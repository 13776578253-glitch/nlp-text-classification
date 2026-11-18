"""GPU telemetry, padding statistics, Chrome trace parsing, and time-budget helpers for S4."""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from threading import Event, Thread

import numpy as np

TRAIN_ROWS = 43_211
VAL_ROWS = 5_416
DEFAULT_CAP = 64

_KERNEL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bnb_nf4_dequant_or_gemm", (
        "kdequantize", "dequantize", "kgemm_4bit", "gemv_4bit", "bitsandbytes", "nf4",
    )),
    ("causal_conv1d", ("causal_conv1d",)),
    ("fla_gated_delta_rule", (
        "chunk_gated_delta", "fused_recurrent", "recompute_w_u", "prepare_wy",
        "solve_tril", "cumsum", "chunk_local_cumsum", "chunk_scaled_dot",
        "chunk_bwd", "chunk_fwd", "gated_delta", "l2norm",
    )),
    ("fla_norm", ("fusedrmsnormgated", "rms_norm_gated", "layernorm_fwd", "layernorm_bwd")),
    ("sdpa_attention", ("flash_", "fmha", "sdpa", "efficient_attention", "attention_forward")),
    ("cublas_gemm", ("nvjet", "cutlass", "cublas", "gemm", "gemmk", "ampere_", "ada_", "sm89_")),
    ("optimizer_adamw", ("adam", "multi_tensor")),
    ("elementwise_cast_norm_act", (
        "elementwise", "vectorized", "unrolled", "reduce_kernel", "indexselect",
        "catarray", "directcopy", "fillfunctor", "distribution", "dropout",
        "softmax", "sigmoid", "silu", "softplus", "cross_entropy", "binary_cross_entropy",
        "rsqrt", "pow", "cumsum_kernel", "arange", "copy_", "cast",
    )),
)


class GpuTelemetryMonitor:
    """Sample nvidia-smi utilization, VRAM, power, SM clock, and temperature in the background."""

    QUERY = (
        "timestamp,utilization.gpu,utilization.memory,"
        "memory.used,memory.free,power.draw,clocks.sm,temperature.gpu"
    )

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = Event()
        self._thread: Thread | None = None
        self._window_start: float | None = None
        self._window_end: float | None = None

    def _poll_once(self) -> dict | None:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        line = proc.stdout.strip().splitlines()[0]
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 8:
            return None
        return {
            "t_mono": time.perf_counter(),
            "timestamp": parts[0],
            "util_gpu_pct": float(parts[1]),
            "util_mem_pct": float(parts[2]),
            "used_mib": int(float(parts[3])),
            "free_mib": int(float(parts[4])),
            "power_w": float(parts[5]),
            "sm_clock_mhz": int(float(parts[6])),
            "temp_c": int(float(parts[7])),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            row = self._poll_once()
            if row is not None:
                self.samples.append(row)
            self._stop.wait(self.interval)

    def __enter__(self) -> GpuTelemetryMonitor:  # noqa: PYI034
        row = self._poll_once()
        if row is not None:
            self.samples.append(row)
        self._thread = Thread(target=self._run, name="s4-gpu-telemetry", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def mark_window_start(self) -> None:
        self._window_start = time.perf_counter()

    def mark_window_end(self) -> None:
        self._window_end = time.perf_counter()

    def summarize(self) -> dict:
        all_rows = self.samples
        if not all_rows:
            return {"samples": 0, "window_samples": 0}
        if self._window_start is not None and self._window_end is not None:
            win = [r for r in all_rows if self._window_start <= r["t_mono"] <= self._window_end]
        else:
            win = all_rows
        if not win:
            win = all_rows[-1:]
        utils = np.asarray([r["util_gpu_pct"] for r in win], dtype=float)
        mem_utils = np.asarray([r["util_mem_pct"] for r in win], dtype=float)
        powers = np.asarray([r["power_w"] for r in win], dtype=float)
        clocks = np.asarray([r["sm_clock_mhz"] for r in win], dtype=float)
        return {
            "samples": len(all_rows),
            "window_samples": len(win),
            "util_mean_pct": float(utils.mean()),
            "util_median_pct": float(np.median(utils)),
            "util_p95_pct": float(np.percentile(utils, 95)),
            "mem_util_mean_pct": float(mem_utils.mean()),
            "peak_used_mib": int(max(r["used_mib"] for r in all_rows)),
            "min_free_mib": int(min(r["free_mib"] for r in all_rows)),
            "window_peak_used_mib": int(max(r["used_mib"] for r in win)),
            "window_min_free_mib": int(min(r["free_mib"] for r in win)),
            "power_mean_w": float(powers.mean()),
            "power_max_w": float(powers.max()),
            "sm_clock_mean_mhz": float(clocks.mean()),
            "sm_clock_max_mhz": int(clocks.max()),
            "temp_max_c": int(max(r["temp_c"] for r in all_rows)),
        }


def compute_padding_stats(
    lengths: np.ndarray,
    batch_size: int,
    multiple: int,
    strategy: str,
    *,
    cap: int = DEFAULT_CAP,
    effective_batch: int = 16,
) -> dict:
    """Compute exact valid vs padded token counts for a given batching and padding policy."""
    clipped = np.minimum(np.asarray(lengths, dtype=np.int64), cap)
    n = len(clipped)
    if strategy == "static_64":
        ordered = clipped
        widths = [cap] * math.ceil(n / batch_size)
        sizes = [min(batch_size, n - i) for i in range(0, n, batch_size)]
    elif strategy == "dynamic_unsorted":
        ordered = clipped
        widths, sizes = [], []
        for i in range(0, n, batch_size):
            chunk = ordered[i:i + batch_size]
            w = math.ceil(int(chunk.max()) / multiple) * multiple
            widths.append(min(cap, w))
            sizes.append(len(chunk))
    elif strategy == "dynamic_intra_effective_sorted":
        ordered_parts = []
        for i in range(0, n, effective_batch):
            block = clipped[i:i + effective_batch]
            ordered_parts.append(block[np.argsort(block, kind="stable")])
        ordered = np.concatenate(ordered_parts) if ordered_parts else clipped
        widths, sizes = [], []
        for i in range(0, n, effective_batch):
            block = ordered[i:i + effective_batch]
            for j in range(0, len(block), batch_size):
                chunk = block[j:j + batch_size]
                w = math.ceil(int(chunk.max()) / multiple) * multiple
                widths.append(min(cap, w))
                sizes.append(len(chunk))
    elif strategy == "dynamic_global_sorted":
        ordered = clipped[np.argsort(clipped, kind="stable")]
        widths, sizes = [], []
        for i in range(0, n, batch_size):
            chunk = ordered[i:i + batch_size]
            w = math.ceil(int(chunk.max()) / multiple) * multiple
            widths.append(min(cap, w))
            sizes.append(len(chunk))
    else:
        raise ValueError(f"Unknown padding strategy: {strategy}")

    valid_tokens = int(clipped.sum())
    padded_tokens = int(sum(w * s for w, s in zip(widths, sizes, strict=True)))
    hist: dict[int, int] = defaultdict(int)
    for w in widths:
        hist[int(w)] += 1
    return {
        "rows": n,
        "batch_size": batch_size,
        "multiple": multiple,
        "strategy": strategy,
        "micro_batches": len(widths),
        "valid_tokens": valid_tokens,
        "padded_tokens": padded_tokens,
        "valid_token_ratio": valid_tokens / padded_tokens if padded_tokens else 0.0,
        "waste_token_ratio": 1.0 - (valid_tokens / padded_tokens) if padded_tokens else 0.0,
        "mean_valid_len": valid_tokens / n if n else 0.0,
        "mean_padded_len": padded_tokens / n if n else 0.0,
        "width_histogram": dict(sorted(hist.items())),
    }


def merge_intervals(intervals: list[tuple[float, float]]) -> float:
    """Return total duration covered by a union of [start, end] intervals."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def classify_cuda_kernel(name: str, cat: str) -> str:
    """Map a Chrome trace GPU event name/category to a mutually-exclusive bucket."""
    if cat in {"gpu_memcpy", "gpu_memset"}:
        return "memcpy_memset"
    lowered = name.lower()
    for bucket, needles in _KERNEL_RULES:
        if any(needle in lowered for needle in needles):
            return bucket
    return "other"


_TARGET_CATS = ('"cat": "kernel"', '"cat": "gpu_memcpy"', '"cat": "gpu_memset"', '"cat": "cuda_runtime"')


def _iter_chrome_trace_events(trace_path: Path):
    """Stream GPU/CUDA-runtime events from large Chrome trace files with O(1) memory."""
    if trace_path.stat().st_size < 10_000_000:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        yield from payload.get("traceEvents", [])
        return

    in_events = False
    buf: list[str] = []
    with trace_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not in_events:
                if '"traceEvents"' in line:
                    in_events = True
                continue
            if line.startswith("  {"):
                buf = [line]
                continue
            if buf:
                buf.append(line)
                stripped = line.rstrip()
                if stripped in {"  }", "  },"}:
                    block = "".join(buf)
                    buf = []
                    if '"ph": "X"' in block and any(tag in block for tag in _TARGET_CATS):
                        yield json.loads(block.rstrip(" \t\r\n,"))
            elif line.startswith("  ]"):
                break


def parse_chrome_trace(trace_path: Path, wall_seconds: float | None = None) -> dict:
    """Analyze a PyTorch Chrome trace for GPU active time, launch gaps, and top kernels."""
    kernel_intervals: list[tuple[float, float]] = []
    gpu_all_intervals: list[tuple[float, float]] = []
    per_kernel_us: dict[str, list[float]] = defaultdict(list)
    per_kernel_occ_weighted: dict[str, float] = defaultdict(float)
    per_bucket_us: dict[str, float] = defaultdict(float)
    per_bucket_calls: dict[str, int] = defaultdict(int)
    cat_counts: dict[str, int] = defaultdict(int)
    kernel_sum_us = 0.0
    gpu_all_sum_us = 0.0
    occ_weighted_sum = 0.0
    occ_dur_sum = 0.0
    sync_calls = 0
    sync_cpu_us = 0.0

    for event in _iter_chrome_trace_events(trace_path):
        if event.get("ph") != "X":
            continue
        cat = str(event.get("cat", ""))
        dur = float(event.get("dur", 0.0))
        ts = float(event.get("ts", 0.0))
        name = str(event.get("name", ""))
        cat_counts[cat] += 1
        if cat == "cuda_runtime" and (
            "StreamSynchronize" in name or "DeviceSynchronize" in name or "EventSynchronize" in name
        ):
            sync_calls += 1
            sync_cpu_us += dur
        if cat in {"kernel", "gpu_memcpy", "gpu_memset"} and dur > 0:
            interval = (ts, ts + dur)
            gpu_all_intervals.append(interval)
            gpu_all_sum_us += dur
            bucket = classify_cuda_kernel(name, cat)
            per_bucket_us[bucket] += dur
            per_bucket_calls[bucket] += 1
            if cat == "kernel":
                kernel_intervals.append(interval)
                kernel_sum_us += dur
                per_kernel_us[name].append(dur)
                occ = event.get("args", {}).get("est. achieved occupancy %")
                if occ is not None:
                    occ_val = float(occ)
                    per_kernel_occ_weighted[name] += occ_val * dur
                    occ_weighted_sum += occ_val * dur
                    occ_dur_sum += dur

    kernel_active_us = merge_intervals(kernel_intervals)
    gpu_all_active_us = merge_intervals(gpu_all_intervals)
    trace_span_us = (
        max(e for _, e in gpu_all_intervals) - min(s for s, _ in gpu_all_intervals)
        if gpu_all_intervals else 0.0
    )
    ref_us = wall_seconds * 1e6 if wall_seconds and wall_seconds > 0 else trace_span_us
    launch_gap_us = max(0.0, ref_us - gpu_all_active_us)
    top_kernels = sorted(
        (
            {
                "name": name,
                "bucket": classify_cuda_kernel(name, "kernel"),
                "calls": len(durs),
                "total_ms": sum(durs) / 1000.0,
                "mean_us": sum(durs) / len(durs),
                "est_occupancy_pct": (
                    per_kernel_occ_weighted[name] / sum(durs)
                    if name in per_kernel_occ_weighted and sum(durs) > 0 else None
                ),
                "share_of_kernel_sum_pct": (100.0 * sum(durs) / kernel_sum_us)
                if kernel_sum_us else 0.0,
            }
            for name, durs in per_kernel_us.items()
        ),
        key=lambda row: row["total_ms"],
        reverse=True,
    )
    buckets = {
        bucket: {
            "calls": per_bucket_calls[bucket],
            "total_ms": per_bucket_us[bucket] / 1000.0,
            "share_of_gpu_sum_pct": (100.0 * per_bucket_us[bucket] / gpu_all_sum_us)
            if gpu_all_sum_us else 0.0,
        }
        for bucket in sorted(per_bucket_us, key=lambda key: per_bucket_us[key], reverse=True)
    }
    return {
        "trace_path": trace_path.as_posix(),
        "wall_seconds": wall_seconds,
        "trace_span_seconds": trace_span_us / 1e6,
        "kernel_count": len(kernel_intervals),
        "memcpy_memset_count": len(gpu_all_intervals) - len(kernel_intervals),
        "kernel_sum_seconds": kernel_sum_us / 1e6,
        "kernel_active_seconds": kernel_active_us / 1e6,
        "gpu_all_active_seconds": gpu_all_active_us / 1e6,
        "gpu_busy_pct_of_wall": (100.0 * gpu_all_active_us / ref_us) if ref_us else 0.0,
        "launch_gap_seconds": launch_gap_us / 1e6,
        "launch_gap_pct_of_wall": (100.0 * launch_gap_us / ref_us) if ref_us else 0.0,
        "est_occupancy_weighted_pct": (occ_weighted_sum / occ_dur_sum) if occ_dur_sum else None,
        "cuda_sync_calls": sync_calls,
        "cuda_sync_cpu_seconds": sync_cpu_us / 1e6,
        "buckets": buckets,
        "top_kernels": top_kernels[:25],
        "category_counts": dict(cat_counts),
    }


def build_time_budget(
    train_sps: float,
    val_sps: float,
    *,
    compile_seconds: float = 0.0,
    load_seconds: float = 12.0,
    checkpoint_save_seconds: float = 5.0,
) -> dict:
    """Compute optimistic / neutral / conservative total budget across standard schedules."""
    schedules = (
        ("1_seed_5_epochs", 1, 5),
        ("3_seeds_2_epochs", 3, 2),
        ("3_seeds_3_epochs", 3, 3),
        ("3_seeds_5_epochs", 3, 5),
    )
    regimes = {"optimistic": 0.95, "neutral": 1.00, "conservative": 1.15}
    output = {}
    for label, seeds, epochs in schedules:
        train_hours = (TRAIN_ROWS * seeds * epochs) / train_sps / 3600.0
        val_hours = (VAL_ROWS * seeds * epochs) / val_sps / 3600.0
        overhead_hours = (
            seeds * (load_seconds + compile_seconds) + seeds * epochs * checkpoint_save_seconds
        ) / 3600.0
        output[label] = {
            "seeds": seeds,
            "epochs": epochs,
            "train_hours_neutral": train_hours,
            "validation_hours_neutral": val_hours,
            "overhead_hours": overhead_hours,
            "scenarios_hours": {
                name: (train_hours + val_hours) * factor + overhead_hours
                for name, factor in regimes.items()
            },
        }
    return output
