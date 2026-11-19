"""Supervise the S4 CUDA worker from a process with no CUDA context."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("artifacts/batch-b-s4/formal")
CUDA_OOM_EXIT = 75
WATCHDOG_EXIT = 124
VRAM_WARNING_MIB = 2560
VRAM_CRITICAL_MIB = 1024
RETRY_FREE_MIB = 10240
STARTUP_PHASES = {"STARTING", "RESOURCES", "VERIFY_INPUTS", "INPUTS_OK", "LOAD_MODEL"}
VALIDATION_PHASES = {"VALIDATE", "VALIDATION_ROWS"}
FORMAL_WORK = "nlp_text_classification.experiments.batch_b_s4_formal"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Batch B S4 training and bounded CUDA acceptance")
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("preflight", "verify-preflight"):
        commands.add_parser(mode).add_argument("--attempt", required=True)
    for mode in ("train", "resume", "status"):
        command = commands.add_parser(mode)
        command.add_argument("--seed", type=int, choices=(42, 123, 2026), required=True)
        if mode != "status":
            command.add_argument("--preflight", required=True)
    args = parser.parse_args(argv)
    attempt = getattr(args, "attempt", None) or getattr(args, "preflight", None)
    if attempt is not None and (not attempt.isascii() or not attempt.replace("-", "").isalnum()):
        parser.error("attempt must consist of ASCII letters, digits and hyphens")
    return args


@dataclass(frozen=True)
class Limits:
    poll_seconds: float = 2
    startup_seconds: float = 300
    work_seconds: float = 120
    validation_seconds: float = 75
    critical_seconds: float = 15
    terminate_seconds: float = 12
    release_seconds: float = 45


def _record(root, event, **details):
    row = {"time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "event": event, **details}
    print(f"[{row['time']}] WATCHDOG {event} {json.dumps(details)}", flush=True)
    with (root / "watchdog.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def _last_progress(path, pid):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        end = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, end - 16384))
        for line in reversed(stream.read().splitlines()):
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if event.get("pid") == pid:
                return event
    return None


def _board_free():
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=3,
        )
        return int(output.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _stop_worker(worker, seconds):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(worker.pid, sig)
        except ProcessLookupError:
            pass
        try:
            worker.wait(timeout=seconds)
            return True
        except subprocess.TimeoutExpired:
            pass
    return False


def _run_worker(command, root, limits, *, protect_vram):
    worker = subprocess.Popen(command, start_new_session=True)
    try:
        _record(root, "WORKER_STARTED", pid=worker.pid)
        latest = critical_since = None
        phase, changed = "STARTING", time.monotonic()
        while worker.poll() is None:
            progress = _last_progress(root / "progress.jsonl", worker.pid)
            now = time.monotonic()
            if progress is not None and progress != latest:
                latest, changed, phase = progress, now, progress["phase"]
            free = _board_free() if protect_vram else None
            if free is not None and free < VRAM_CRITICAL_MIB:
                critical_since = now if critical_since is None else critical_since
            else:
                critical_since = None
            budget = limits.work_seconds
            if phase in STARTUP_PHASES:
                budget = limits.startup_seconds
            elif phase in VALIDATION_PHASES:
                budget = limits.validation_seconds
            reason = None
            if now - changed >= budget:
                reason = "no_progress"
            elif critical_since is not None and now - critical_since >= limits.critical_seconds:
                reason = "sustained_critical_vram"
            if reason and worker.poll() is None:
                _record(root, "STOPPING_WORKER", pid=worker.pid, reason=reason, phase=phase,
                        last_progress_age_seconds=round(now - changed, 2), free_mib=free)
                released = _stop_worker(worker, limits.terminate_seconds)
                _record(root, "WORKER_RELEASED" if released else "WORKER_NOT_RELEASED",
                        pid=worker.pid, returncode=worker.returncode)
                return WATCHDOG_EXIT, released
            time.sleep(limits.poll_seconds)
        _record(root, "WORKER_EXITED", pid=worker.pid, returncode=worker.returncode)
        return worker.returncode, worker.returncode == CUDA_OOM_EXIT
    except BaseException:
        released = _stop_worker(worker, limits.terminate_seconds)
        _record(root, "SUPERVISOR_INTERRUPTED", pid=worker.pid, released=released)
        raise


def _wait_for_release(seconds, poll_seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        free = _board_free()
        if free is not None and free >= RETRY_FREE_MIB:
            return free
        time.sleep(poll_seconds)
    return None


def _worker_command(args):
    return [sys.executable, "-X", "faulthandler", "-u", "-m", FORMAL_WORK, *args]


def supervise(args, root, limits=Limits()):
    root.mkdir(parents=True, exist_ok=True)
    formal = args[0] in {"train", "resume"}
    code, retryable = _run_worker(_worker_command(args), root, limits, protect_vram=formal)
    if not formal or not retryable:
        return code
    available = _wait_for_release(limits.release_seconds, limits.poll_seconds)
    if available is None:
        _record(root, "RETRY_SKIPPED", reason="VRAM release unconfirmed", required_mib=RETRY_FREE_MIB)
        return code
    mode = "resume" if any(root.glob("checkpoints/step-????????/checkpoint.json")) else args[0]
    _record(root, "RETRY", mode=mode, free_mib=available, previous_returncode=code)
    code, _ = _run_worker(_worker_command([mode, *args[1:]]), root, limits, protect_vram=True)
    return code


def main():
    args = parse_args()
    if args.mode == "status":
        os.execv(sys.executable, _worker_command(sys.argv[1:]))
    if args.mode in {"train", "resume"}:
        root = ROOT / "runs" / f"seed-{args.seed}"
    else:
        root = ROOT / "preflight" / args.attempt
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGHUP, _interrupt)
    try:
        code = supervise(sys.argv[1:], root)
    except KeyboardInterrupt:
        code = 130
    sys.exit(code if code >= 0 else 128 - code)


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    main()
