"""Protect the workstation while Batch A runs in the background."""

from __future__ import annotations

import _thread
import ctypes
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Thread

GIB = 1024 ** 3
START_FREE = 8 * GIB
PAUSE_FREE = 6 * GIB
WINDOWS_PROCESS_LIMIT = 4 * GIB
CPU_THREADS = 2


@dataclass(frozen=True)
class AvailableMemory:
    physical: int
    commit: int

    def below(self, threshold: int) -> bool:
        return min(self.physical, self.commit) < threshold

    def display(self) -> dict:
        return {"free_physical_gib": round(self.physical / GIB, 2),
                "free_commit_gib": round(self.commit / GIB, 2)}


def memory_available() -> AvailableMemory:
    if sys.platform == "win32":
        class Performance(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32),
                *((field, ctypes.c_size_t) for field in (
                    "commit_total", "commit_limit", "commit_peak", "physical_total",
                    "physical_available", "system_cache", "kernel_total", "kernel_paged",
                    "kernel_non_paged", "page_size",
                )),
                ("handles", ctypes.c_uint32), ("processes", ctypes.c_uint32),
                ("threads", ctypes.c_uint32),
            ]

        result = Performance()
        result.length = ctypes.sizeof(result)
        api = ctypes.WinDLL("psapi", use_last_error=True)
        if not api.GetPerformanceInfo(ctypes.byref(result), result.length):
            raise ctypes.WinError(ctypes.get_last_error())
        return AvailableMemory(
            result.physical_available * result.page_size,
            (result.commit_limit - result.commit_total) * result.page_size,
        )

    with open("/proc/meminfo", encoding="ascii") as stream:
        fields = {key: int(value.split()[0]) * 1024
                  for key, _, value in (line.partition(":") for line in stream) if value}
    available = fields["MemAvailable"]
    return AvailableMemory(available, available + fields.get("SwapFree", 0))


def configure_compute_threads():
    """Set the process environment before NumPy, sklearn and torch are loaded."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(CPU_THREADS)


def require_headroom() -> AvailableMemory:
    current = memory_available()
    if current.below(START_FREE):
        raise RuntimeError(
            f"Batch paused: {current.physical / GIB:.2f} GiB physical and "
            f"{current.commit / GIB:.2f} GiB commit headroom; startup needs 8 GiB of both"
        )
    return current


def resource_status() -> dict:
    current = memory_available()
    return {**current.display(), "can_start": not current.below(START_FREE),
            "startup_min_gib": START_FREE // GIB, "pause_below_gib": PAUSE_FREE // GIB,
            "windows_process_commit_limit_gib": WINDOWS_PROCESS_LIMIT // GIB,
            "cpu_threads": CPU_THREADS}


def set_windows_process_limit(limit: int, *, lower_priority: bool = True):
    """Attach a process-wide private commit ceiling to the dedicated Batch A CLI."""
    if sys.platform != "win32":
        return

    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("per_process_time", ctypes.c_int64), ("per_job_time", ctypes.c_int64),
            ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
            ("max_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t), ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(field, ctypes.c_uint64) for field in (
            "read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes"
        )]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", BasicLimits), ("io", IoCounters), ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t),
            ("peak_job", ctypes.c_size_t),
        ]

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    api.CreateJobObjectW.restype = wintypes.HANDLE
    api.GetCurrentProcess.restype = wintypes.HANDLE
    api.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
    ]
    api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    api.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        values = ExtendedLimits()
        values.basic.flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        values.process_memory = limit
        process = api.GetCurrentProcess()
        if not api.SetInformationJobObject(handle, 9, ctypes.byref(values), ctypes.sizeof(values)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not api.AssignProcessToJobObject(handle, process):
            raise ctypes.WinError(ctypes.get_last_error())
        if lower_priority and not api.SetPriorityClass(process, 0x4000):  # BELOW_NORMAL_PRIORITY_CLASS
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        api.CloseHandle(handle)


@contextmanager
def protect(progress, *, process_limit=WINDOWS_PROCESS_LIMIT):
    current = require_headroom()
    set_windows_process_limit(process_limit)
    progress.emit("RESOURCES", "memory guard active", **current.display(),
                  windows_commit_limit_gib=process_limit // GIB, cpu_threads=CPU_THREADS)
    done, low_memory = Event(), Event()

    def watch():
        while not done.wait(1):
            snapshot = memory_available()
            if snapshot.below(PAUSE_FREE):
                low_memory.set()
                progress.emit("PAUSED_MEMORY", "system RAM or commit headroom fell below 6 GiB",
                              **snapshot.display())
                _thread.interrupt_main()
                if not done.wait(3):
                    os._exit(75)
                return

    watcher = Thread(target=watch, name="batch-memory-guard", daemon=True)
    watcher.start()
    try:
        yield
    except KeyboardInterrupt:
        if low_memory.is_set():
            raise RuntimeError("Batch paused to preserve workstation memory") from None
        raise
    finally:
        done.set()
