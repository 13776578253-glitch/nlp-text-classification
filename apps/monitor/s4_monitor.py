"""Read-only live monitor for the Batch B S4 formal training runs.

The monitor tails the run's JSONL journals and samples the GPU from the host.
It never imports torch, never creates a CUDA context, and writes only under
``artifacts/batch-b-s4/formal/monitor/``. Nothing inside a run directory is ever
modified, so the hash-bound training evidence stays untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO / "artifacts" / "batch-b-s4" / "formal"
RUNS_ROOT = FORMAL_ROOT / "runs"
MONITOR_ROOT = FORMAL_ROOT / "monitor"
INDEX = Path(__file__).with_name("index.html")

VRAM_WARNING_MIB = 2560
VRAM_CRITICAL_MIB = 1024
VRAM_RETRY_MIB = 10240

STARTUP_PHASES = {"STARTING", "RESOURCES", "VERIFY_INPUTS", "INPUTS_OK", "LOAD_MODEL"}
VALIDATION_PHASES = {"VALIDATE", "VALIDATION_ROWS"}
NOTABLE_PHASES = {"TRAIN", "VALIDATE", "VALIDATION_ROWS", "EPOCH", "CHECKPOINT"}

EVENT_KEYS = (
    "time",
    "phase",
    "message",
    "global_step",
    "loss",
    "gradient_norm",
    "learning_rates",
    "device_min_free_mib",
    "macro_f1",
    "threshold",
    "train_loss",
    "validation_seconds",
    "elapsed_seconds",
    "stale_epochs",
    "recent_free_mib",
    "warning_mib",
    "error",
    "pid",
    "best",
    "epochs",
    "min_free_mib",
    "reproduced_scores",
    "optimizer_step",
    "scheduler_step",
    "parameter_states",
)

DEFAULT_PLAN = {
    "effective_batch_size": 16,
    "updates_per_epoch": 2701,
    "tail_rows": 11,
    "total_updates": 13505,
    "warmup_updates": 811,
    "epochs": 5,
    "checkpoint_interval": 500,
}

RECOMMENDED = REPO / "reports" / "batch-b-s4" / "performance" / "recommended-training.json"
_LABEL_ORDER: list[str] | None = None


def label_order() -> list[str]:
    global _LABEL_ORDER
    if _LABEL_ORDER is None:
        recipe = _read_json(RECOMMENDED)
        labels = recipe.get("label_order") if isinstance(recipe, dict) else None
        _LABEL_ORDER = [str(label) for label in labels] if isinstance(labels, list) else []
    return _LABEL_ORDER

STATUS_LABELS = {
    "NO_RUN": "Idle",
    "STARTING": "Starting",
    "RUNNING": "Training",
    "VALIDATING": "Validating",
    "STALLED": "No progress",
    "HALTED": "Halted",
    "STOPPING": "Stopping",
    "RECOVERING": "Recovering",
    "EXITED": "Exited",
    "COMPLETE": "Complete",
}

_UPDATE_RE = re.compile(r"Epoch (\d+), update (\d+)/(\d+)")
_SEED_RE = re.compile(r"seed-(\d+)")
_CHECKPOINT_RE = re.compile(r"step-\d{8}")


def _iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds")


def _event_time(row: dict | None) -> float | None:
    if not row:
        return None
    try:
        return datetime.fromisoformat(str(row.get("time"))).timestamp()
    except (TypeError, ValueError):
        return None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _number(raw, cast):
    try:
        return cast(float(raw))
    except (TypeError, ValueError):
        return None


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


class JsonlTail:
    """Incrementally parse an append-only JSONL journal, tolerating partial lines."""

    def __init__(self, path: Path, limit: int = 20000) -> None:
        self.path = path
        self.limit = limit
        self.offset = 0
        self.pending = b""
        self.rows: list[dict] = []

    def poll(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:
            self.offset = 0
            self.pending = b""
            self.rows = []
        if size == self.offset:
            return []
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
        except OSError:
            return []
        self.offset += len(data)
        data = self.pending + data
        lines = data.split(b"\n")
        self.pending = lines.pop()
        fresh = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                fresh.append(row)
        if fresh:
            self.rows.extend(fresh)
            if len(self.rows) > self.limit:
                self.rows = self.rows[-self.limit:]
        return fresh


class GpuSampler(threading.Thread):
    """Sample host-level GPU counters without touching the training process."""

    FIELDS = (
        "utilization.gpu",
        "memory.used",
        "memory.free",
        "memory.total",
        "power.draw",
        "temperature.gpu",
        "clocks.sm",
    )

    def __init__(self, interval: float = 2.0) -> None:
        super().__init__(name="s4-gpu-sampler", daemon=True)
        self.interval = interval
        self.samples: deque[dict] = deque(maxlen=7200)
        self.error: str | None = None
        self._stop = threading.Event()

    def poll(self) -> dict | None:
        query = ",".join(self.FIELDS)
        command = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ]
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return None
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "").strip()
            self.error = detail[:200] or f"nvidia-smi exit {done.returncode}"
            return None
        lines = (done.stdout or "").strip().splitlines()
        if not lines:
            self.error = "nvidia-smi returned no rows"
            return None
        parts = [item.strip() for item in lines[0].split(",")]
        if len(parts) != len(self.FIELDS):
            self.error = f"unexpected nvidia-smi row: {lines[0][:160]}"
            return None
        self.error = None
        return {
            "t": time.time(),
            "util": _number(parts[0], int),
            "used": _number(parts[1], int),
            "free": _number(parts[2], int),
            "total": _number(parts[3], int),
            "power": _number(parts[4], float),
            "temp": _number(parts[5], int),
            "clock": _number(parts[6], int),
        }

    def sample_once(self) -> dict | None:
        row = self.poll()
        if row is not None:
            self.samples.append(row)
        return row

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample_once()

    def stop(self) -> None:
        self._stop.set()


def latest_run_dir() -> Path | None:
    if not RUNS_ROOT.is_dir():
        return None
    runs = [item for item in RUNS_ROOT.iterdir() if item.is_dir() and _SEED_RE.fullmatch(item.name)]
    if not runs:
        return None
    return max(runs, key=lambda item: _mtime(item / "progress.jsonl"))


def plan_for(run_dir: Path) -> dict:
    plan = dict(DEFAULT_PLAN)
    manifest = _read_json(run_dir / "run.json")
    if isinstance(manifest, dict):
        identity = manifest.get("identity") or {}
        for key, value in (identity.get("plan") or {}).items():
            if key in plan and isinstance(value, (int, float)):
                plan[key] = int(value)
        training = (manifest.get("protocol") or {}).get("training") or {}
        if training.get("epochs"):
            plan["epochs"] = int(training["epochs"])
        if identity.get("checkpoint_interval"):
            plan["checkpoint_interval"] = int(identity["checkpoint_interval"])
    if plan["updates_per_epoch"]:
        plan["epochs"] = max(1, round(plan["total_updates"] / plan["updates_per_epoch"]))
    return plan


def latest_checkpoint(run_dir: Path) -> Path | None:
    folder = run_dir / "checkpoints"
    if not folder.is_dir():
        return None
    checkpoints = [item for item in folder.iterdir() if item.is_dir() and _CHECKPOINT_RE.fullmatch(item.name)]
    return max(checkpoints, default=None)


def latest_metrics_path(run_dir: Path) -> Path | None:
    folder = run_dir / "checkpoints"
    if not folder.is_dir():
        return None
    checkpoints = sorted(
        (
            item
            for item in folder.iterdir()
            if item.is_dir() and _CHECKPOINT_RE.fullmatch(item.name)
        ),
        reverse=True,
    )
    for checkpoint in checkpoints:
        path = checkpoint / "metrics_validation.json"
        if path.is_file():
            return path
    return None


def phase_budget(phase: str | None) -> int:
    if phase in STARTUP_PHASES:
        return 300
    if phase in VALIDATION_PHASES:
        return 75
    return 120


def step_rate(train_rows: list[dict], window: int = 10) -> float | None:
    rows = [row for row in train_rows if _event_time(row) is not None][-window:]
    if len(rows) < 2:
        return None
    first_step = rows[0].get("global_step")
    last_step = rows[-1].get("global_step")
    if first_step is None or last_step is None:
        return None
    delta_steps = float(last_step) - float(first_step)
    delta_time = _event_time(rows[-1]) - _event_time(rows[0])
    if delta_steps <= 0 or delta_time <= 0:
        return None
    return delta_steps / delta_time


def estimate_eta(
    step: int,
    batch: int,
    current_epoch: int,
    plan: dict,
    epoch_rows: list[dict],
    rate: float | None,
    stop_reason: str | None,
) -> dict:
    if stop_reason:
        return {"seconds": 0.0, "confidence": "done", "note": "Training finished"}
    per_epoch = max(1, int(plan["updates_per_epoch"]))
    epochs_cap = max(1, int(plan["epochs"]))
    last_wall = None
    if epoch_rows:
        last_wall = _finite(epoch_rows[-1].get("elapsed_seconds"))
        if last_wall is not None and last_wall <= 0:
            last_wall = None
    if last_wall is not None and rate:
        remaining_epochs = max(0, epochs_cap - current_epoch)
        seconds = max(0, per_epoch - batch) / rate + remaining_epochs * last_wall
        return {"seconds": seconds, "confidence": "epoch",
                "note": "Extrapolated from the first measured epoch"}
    if rate:
        seconds = max(0, int(plan["total_updates"]) - step) / rate
        remaining_validations = max(0, epochs_cap - len(epoch_rows))
        seconds += remaining_validations * 45.0
        return {"seconds": seconds, "confidence": "coarse",
                "note": "Coarse estimate from the recent step rate"}
    return {"seconds": None, "confidence": "unknown",
            "note": "Waiting for enough step events"}


def _compact(row: dict, keys) -> dict:
    return {key: row[key] for key in keys if key in row}


def _status(rows: list[dict], watchdog: list[dict], now: float, run_dir: Path) -> dict:
    if not rows and not watchdog:
        return {"status": "NO_RUN", "phase": None, "budget": None, "age": None, "remaining": None}
    if any(row.get("phase") == "COMPLETE" for row in rows):
        last = rows[-1]
        return {
            "status": "COMPLETE",
            "phase": last.get("phase"),
            "budget": None,
            "age": None,
            "remaining": None,
        }
    last_wd = watchdog[-1] if watchdog else None
    last_wd_event = last_wd.get("event") if last_wd else None
    if last_wd_event in {"STOPPING_WORKER", "WORKER_RELEASED"}:
        return {
            "status": "STOPPING",
            "phase": last_wd.get("phase"),
            "budget": None,
            "age": None,
            "remaining": None,
        }
    if last_wd_event == "RETRY":
        return {"status": "RECOVERING", "phase": None, "budget": None, "age": None, "remaining": None}
    if last_wd_event in {
        "WORKER_EXITED",
        "RETRY_SKIPPED",
        "WORKER_NOT_RELEASED",
        "SUPERVISOR_INTERRUPTED",
    }:
        return {"status": "EXITED", "phase": None, "budget": None, "age": None, "remaining": None}
    last = rows[-1] if rows else None
    if last is not None and last.get("phase") == "HALTED":
        return {"status": "HALTED", "phase": "HALTED", "budget": None, "age": None, "remaining": None}
    phase = last.get("phase") if last else "STARTING"
    progress_time = _event_time(last) if last else None
    started_time = _event_time(last_wd) if last_wd_event == "WORKER_STARTED" else None
    candidates = [
        value
        for value in (progress_time, started_time, _mtime(run_dir / "progress.jsonl"))
        if value
    ]
    baseline = max(candidates) if candidates else now
    budget = phase_budget(phase)
    age = max(0.0, now - baseline)
    if age > budget:
        return {
            "status": "STALLED",
            "phase": phase,
            "budget": budget,
            "age": round(age, 1),
            "remaining": round(budget - age, 1),
        }
    if phase in VALIDATION_PHASES:
        status = "VALIDATING"
    elif phase in STARTUP_PHASES:
        status = "STARTING"
    else:
        status = "RUNNING"
    return {
        "status": status,
        "phase": phase,
        "budget": budget,
        "age": round(age, 1),
        "remaining": round(budget - age, 1),
    }


class RunFeed:
    def __init__(self, run_dir: Path | None, *, gpu: GpuSampler | None = None, follow: bool = False):
        self.gpu = gpu
        self.follow = follow
        self.lock = threading.Lock()
        self.run_dir: Path | None = None
        self.progress: JsonlTail | None = None
        self.watchdog: JsonlTail | None = None
        self._metrics_key: tuple | None = None
        self._metrics: dict | None = None
        self._set_run(run_dir)

    def _set_run(self, run_dir: Path | None) -> None:
        self.run_dir = run_dir
        self._metrics_key = None
        self._metrics = None
        if run_dir is None:
            self.progress = None
            self.watchdog = None
            return
        self.progress = JsonlTail(run_dir / "progress.jsonl")
        self.watchdog = JsonlTail(run_dir / "watchdog.jsonl")

    def _maybe_follow(self) -> None:
        if not self.follow:
            return
        candidate = latest_run_dir()
        if candidate is None or candidate == self.run_dir:
            return
        if self.run_dir is None:
            self._set_run(candidate)
            return
        completed = any(
            row.get("phase") == "COMPLETE" for row in (self.progress.rows if self.progress else [])
        )
        if completed or not (self.run_dir / "progress.jsonl").exists():
            self._set_run(candidate)

    def snapshot(self) -> dict:
        with self.lock:
            self._maybe_follow()
            if self.progress is not None:
                self.progress.poll()
            if self.watchdog is not None:
                self.watchdog.poll()
            return self._snapshot(time.time())

    def _gpu_view(self, now: float) -> dict:
        samples = list(self.gpu.samples) if self.gpu is not None else []
        recent = samples[-300:]
        current = samples[-1] if samples else None
        free_values = [item["free"] for item in samples if item.get("free") is not None]
        return {
            "samples": recent,
            "current": current,
            "min_free_mib": min(free_values) if free_values else None,
            "total_mib": current.get("total") if current else None,
            "warn_mib": VRAM_WARNING_MIB,
            "critical_mib": VRAM_CRITICAL_MIB,
            "retry_mib": VRAM_RETRY_MIB,
            "error": self.gpu.error if self.gpu is not None else "disabled",
            "sampled_at": current["t"] if current else None,
            "now": now,
        }

    def _snapshot(self, now: float) -> dict:
        if self.run_dir is None:
            return {
                "generated_at": _iso(now),
                "server_now": now,
                "status": "NO_RUN",
                "status_label": STATUS_LABELS["NO_RUN"],
                "health": {"status": "NO_RUN", "phase": None, "budget": None, "age": None,
                           "remaining": None},
                "run": None,
                "plan": dict(DEFAULT_PLAN),
                "label_order": label_order(),
                "progress": None,
                "best": None,
                "stale_epochs": 0,
                "epochs": [],
                "metrics": None,
                "metrics_step": None,
                "vram": self._gpu_view(now),
                "watchdog": {"events": [], "retries": 0, "worker_pid": None, "last_event": None},
                "counters": {"warnings": 0, "halts": 0, "resumes": 0, "checkpoints": 0},
                "checkpoint": None,
                "events": [],
                "notable_events": [],
                "train_history": [],
                "eta": {"seconds": None, "confidence": "unknown",
                        "note": "Waiting for training to start"},
                "disk": _disk_view(),
                "alerts": [],
                "run_files": {},
            }
        run_dir = self.run_dir
        rows = self.progress.rows if self.progress else []
        watchdog = self.watchdog.rows if self.watchdog else []
        plan = plan_for(run_dir)
        health = _status(rows, watchdog, now, run_dir)
        status = health["status"]
        train_rows = [row for row in rows if row.get("phase") == "TRAIN"]
        epoch_rows = [row for row in rows if row.get("phase") == "EPOCH"]
        last_train = train_rows[-1] if train_rows else None
        last_row = rows[-1] if rows else None
        parsed = None
        if last_train is not None:
            match = _UPDATE_RE.search(str(last_train.get("message", "")))
            if match:
                parsed = tuple(int(value) for value in match.groups())
        completed = len(epoch_rows)
        if parsed:
            current_epoch = parsed[0]
            batch = parsed[1]
            updates_per_epoch = parsed[2]
        else:
            current_epoch = min(plan["epochs"], completed + (0 if completed >= plan["epochs"] else 1))
            batch = 0
            updates_per_epoch = plan["updates_per_epoch"]
        step = 0
        for row in rows:
            value = row.get("global_step")
            if isinstance(value, (int, float)):
                step = max(step, int(value))
        checkpoint_path = latest_checkpoint(run_dir)
        checkpoint_manifest = _read_json(checkpoint_path / "checkpoint.json") if checkpoint_path else None
        checkpoint_state = (checkpoint_manifest or {}).get("state") or {}
        if checkpoint_state.get("global_step") is not None:
            step = max(step, int(checkpoint_state["global_step"]))
        stop_reason = checkpoint_state.get("stop_reason")
        best = None
        if epoch_rows:
            best = epoch_rows[-1].get("best")
        if best is None:
            best = checkpoint_state.get("best")
        stale_epochs = 0
        if epoch_rows:
            stale_epochs = int(epoch_rows[-1].get("stale_epochs") or 0)
        elif checkpoint_state.get("stale_epochs") is not None:
            stale_epochs = int(checkpoint_state["stale_epochs"])
        total = max(1, int(plan["total_updates"]))
        fraction = min(1.0, step / total)
        epoch_fraction = min(1.0, batch / max(1, updates_per_epoch))
        rate = step_rate(train_rows)
        eta = estimate_eta(step, batch, current_epoch, plan, epoch_rows, rate, stop_reason)
        last_time = _event_time(last_row) if last_row else None
        first_time = _event_time(rows[0]) if rows else None
        started_at = first_time or _mtime(run_dir / "progress.jsonl") or now
        vram_train_min = None
        for row in train_rows:
            value = row.get("device_min_free_mib")
            if isinstance(value, (int, float)):
                vram_train_min = value if vram_train_min is None else min(vram_train_min, value)
        warnings = [row for row in rows if row.get("phase") == "VRAM_WARNING"]
        halts = [row for row in rows if row.get("phase") == "HALTED"]
        resumes = [row for row in rows if row.get("phase") == "RESUMED"]
        checkpoints = [row for row in rows if row.get("phase") == "CHECKPOINT"]
        metrics = None
        metrics_step = None
        metrics_path = latest_metrics_path(run_dir)
        if metrics_path is not None:
            key = (str(metrics_path), _mtime(metrics_path))
            if key != self._metrics_key:
                self._metrics = _read_json(metrics_path)
                self._metrics_key = key
            metrics = self._metrics
            match = _CHECKPOINT_RE.fullmatch(metrics_path.parent.name)
            if match:
                metrics_step = int(metrics_path.parent.name.split("-")[1])
        watchdog_events = watchdog[-40:]
        worker_pid = None
        for event in reversed(watchdog):
            if event.get("event") == "WORKER_STARTED":
                worker_pid = event.get("pid")
                break
        retries = sum(1 for event in watchdog if event.get("event") == "RETRY")
        stopping = None
        for event in reversed(watchdog):
            if event.get("event") == "STOPPING_WORKER":
                stopping = event
                break
        epochs = []
        for row in epoch_rows:
            entry = _compact(
                row,
                (
                    "time",
                    "epoch",
                    "global_step",
                    "macro_f1",
                    "threshold",
                    "train_loss",
                    "validation_seconds",
                    "elapsed_seconds",
                    "stale_epochs",
                ),
            )
            entry["is_best"] = bool(best and entry.get("epoch") == best.get("epoch"))
            epochs.append(entry)
        history = []
        for row in train_rows[-400:]:
            history.append(
                {
                    "step": row.get("global_step"),
                    "loss": _finite(row.get("loss")),
                    "grad": _finite(row.get("gradient_norm")),
                    "lr_rank": _first(_finite_list(row.get("learning_rates"), 0)),
                    "lr_head": _first(_finite_list(row.get("learning_rates"), 1)),
                    "valid_tokens": row.get("valid_tokens"),
                    "padded_tokens": row.get("padded_tokens"),
                    "vram_min": row.get("device_min_free_mib"),
                    "t": _event_time(row),
                }
            )
        events = [_compact(row, EVENT_KEYS) for row in rows[-60:]]
        notable_events = [
            _compact(row, EVENT_KEYS)
            for row in rows
            if row.get("phase") not in NOTABLE_PHASES
        ][-40:]
        alerts = _alerts(status, last_row, watchdog, warnings, halts, now)
        return {
            "generated_at": _iso(now),
            "server_now": now,
            "status": status,
            "status_label": STATUS_LABELS.get(status, status),
            "health": health,
            "run": {
                "seed": _seed_of(run_dir),
                "dir": str(run_dir),
                "mode": last_row.get("mode") if last_row else None,
                "started_at": _iso(started_at),
                "elapsed_seconds": max(0.0, now - started_at),
            },
            "plan": plan,
            "label_order": label_order(),
            "progress": {
                "global_step": step,
                "epoch": current_epoch,
                "batch": batch,
                "updates_per_epoch": updates_per_epoch,
                "fraction": fraction,
                "epoch_fraction": epoch_fraction,
                "loss": _finite(last_train.get("loss")) if last_train else None,
                "loss_epoch_mean": _finite(epoch_rows[-1].get("train_loss")) if epoch_rows else None,
                "learning_rates": _finite_list(last_train.get("learning_rates"), None) if last_train else [],
                "gradient_norm": _finite(last_train.get("gradient_norm")) if last_train else None,
                "gradient_checkpointing": last_train.get("gradient_checkpointing") if last_train else None,
                "vram_train_min_mib": vram_train_min,
                "valid_tokens": last_train.get("valid_tokens") if last_train else None,
                "padded_tokens": last_train.get("padded_tokens") if last_train else None,
                "rate_steps_per_second": rate,
                "phase": last_row.get("phase") if last_row else None,
                "message": last_row.get("message") if last_row else None,
                "last_event_age_seconds": round(now - last_time, 1) if last_time else None,
            },
            "best": best,
            "stale_epochs": stale_epochs,
            "epochs": epochs,
            "metrics": metrics,
            "metrics_step": metrics_step,
            "vram": self._gpu_view(now),
            "watchdog": {
                "events": watchdog_events,
                "retries": retries,
                "worker_pid": worker_pid,
                "last_event": watchdog[-1].get("event") if watchdog else None,
                "stopping": stopping,
            },
            "counters": {
                "warnings": len(warnings),
                "halts": len(halts),
                "resumes": len(resumes),
                "checkpoints": len(checkpoints),
            },
            "checkpoint": {
                "path": str(checkpoint_path) if checkpoint_path else None,
                "step": checkpoint_state.get("global_step"),
                "age_seconds": round(now - _mtime(checkpoint_path / "checkpoint.json"), 1)
                if checkpoint_path
                else None,
            },
            "events": events,
            "notable_events": notable_events,
            "train_history": history,
            "eta": eta,
            "disk": _disk_view(),
            "alerts": alerts,
            "run_files": {
                "progress": str(run_dir / "progress.jsonl"),
                "watchdog": str(run_dir / "watchdog.jsonl"),
                "run": str(run_dir / "run.json"),
                "checkpoints": str(run_dir / "checkpoints"),
            },
        }


def _first(values: list) -> float | None:
    return values[0] if values else None


def _finite_list(value, index: int | None) -> list:
    if not isinstance(value, list):
        return []
    numbers = [_finite(item) for item in value]
    numbers = [item for item in numbers if item is not None]
    if index is None:
        return numbers
    return numbers[index:index + 1]


def _seed_of(run_dir: Path) -> int | None:
    match = _SEED_RE.search(run_dir.name)
    return int(match.group(1)) if match else None


def _disk_view() -> dict:
    try:
        usage = shutil.disk_usage(str(REPO.anchor))
    except OSError:
        return {"free_gib": None, "total_gib": None}
    return {
        "free_gib": usage.free / (1 << 30),
        "total_gib": usage.total / (1 << 30),
    }


def _alerts(
    status: str,
    last_row: dict | None,
    watchdog: list[dict],
    warnings: list[dict],
    halts: list[dict],
    now: float,
) -> list[dict]:
    alerts = []
    if status == "STALLED":
        phase = last_row.get("phase") if last_row else "STARTING"
        alerts.append(
            {
                "level": "critical",
                "text": f"No new event for more than {phase_budget(phase)} s; "
                f"the watchdog will terminate the worker.",
            }
        )
    if halts and (last_row is None or last_row.get("phase") == "HALTED"):
        last_halt = halts[-1]
        age = now - (_event_time(last_halt) or now)
        if age < 1800:
            alerts.append(
                {
                    "level": "critical",
                    "text": f"Worker reported HALTED: {last_halt.get('error', 'unknown')}; "
                    f"last seen {int(age)} s ago.",
                }
            )
    if warnings:
        last_warning = warnings[-1]
        age = now - (_event_time(last_warning) or now)
        if age < 600:
            alerts.append(
                {
                    "level": "warning",
                    "text": f"VRAM free dropped below {VRAM_WARNING_MIB} MiB: "
                    f"{last_warning.get('recent_free_mib')} MiB, {int(age)} s ago.",
                }
            )
    if status == "RECOVERING":
        alerts.append(
            {
                "level": "warning",
                "text": f"Worker exited; waiting for {VRAM_RETRY_MIB} MiB free "
                f"before one automatic retry.",
            }
        )
    if watchdog:
        last = watchdog[-1]
        if last.get("event") == "STOPPING_WORKER":
            alerts.append(
                {
                    "level": "critical",
                    "text": f"Watchdog is terminating the worker: {last.get('reason')} "
                    f"(free {last.get('free_mib')} MiB).",
                }
            )
        if last.get("event") == "RETRY_SKIPPED":
            alerts.append(
                {
                    "level": "critical",
                    "text": f"Automatic retry skipped: {last.get('reason')}. "
                    f"Manual action required.",
                }
            )
        if last.get("event") == "WORKER_NOT_RELEASED":
            alerts.append(
                {
                    "level": "critical",
                    "text": "Worker did not exit; the watchdog escalated to SIGKILL. "
                    "Check the process and VRAM.",
                }
            )
    return alerts


def digest_text(snapshot: dict) -> str:
    status = snapshot.get("status", "NO_RUN")
    label = STATUS_LABELS.get(status, status)
    if status == "NO_RUN" or snapshot.get("run") is None:
        return f"S4 monitor | {label} | no training run found"
    run = snapshot["run"]
    plan = snapshot["plan"]
    progress = snapshot["progress"] or {}
    eta = snapshot.get("eta") or {}
    vram = snapshot.get("vram") or {}
    watchdog = snapshot.get("watchdog") or {}
    counters = snapshot.get("counters") or {}
    current = vram.get("current") or {}
    best = snapshot.get("best")
    rate = progress.get("rate_steps_per_second")
    loss = _fmt(progress.get("loss"), 4)
    grad = _fmt(progress.get("gradient_norm"), 4)
    lrs = progress.get("learning_rates") or []
    lr_text = "/".join(f"{value:.2e}" for value in lrs[:2]) if lrs else "—"
    free = current.get("free")
    free_text = f"{free / 1024:.1f} GiB" if free is not None else "—"
    train_min = progress.get("vram_train_min_mib")
    train_min_text = f"{train_min / 1024:.1f} GiB" if train_min is not None else "—"
    best_text = "—"
    if best:
        best_text = (
            f"{_fmt(best.get('macro_f1'), 4)} @ epoch {best.get('epoch')} "
            f"(threshold {_fmt(best.get('threshold'), 2)})"
        )
    eta_text = _duration(eta.get("seconds"))
    if eta.get("seconds") is not None and eta.get("confidence") == "coarse":
        eta_text += " (coarse)"
    checkpoint = snapshot.get("checkpoint") or {}
    ckpt_text = "—"
    if checkpoint.get("step") is not None:
        ckpt_text = f"step-{int(checkpoint['step']):08d} ({_duration(checkpoint.get('age_seconds'))} ago)"
    health = snapshot.get("health") or {}
    age = progress.get("last_event_age_seconds")
    lines = [
        (
            f"S4 seed {run.get('seed')} | {label} | phase {progress.get('phase')} | "
            f"step {progress.get('global_step', 0):,}/{plan.get('total_updates', 0):,} "
            f"({100 * (progress.get('fraction') or 0):.1f}%) | "
            f"epoch {progress.get('epoch')}/{plan.get('epochs')} "
            f"batch {progress.get('batch')}/{progress.get('updates_per_epoch')}"
        ),
        (
            f"loss {loss} | grad {grad} | lr {lr_text} | rate "
            f"{f'{rate:.2f} step/s' if rate else '—'} | vram free {free_text} "
            f"(train min {train_min_text})"
        ),
        (
            f"best {best_text} | stale {snapshot.get('stale_epochs', 0)} | ckpt {ckpt_text} | "
            f"eta {eta_text} | last event {_duration(age)} ago"
        ),
        (
            f"warnings {counters.get('warnings', 0)} | halts {counters.get('halts', 0)} | "
            f"resumes {counters.get('resumes', 0)} | "
            f"watchdog retries {watchdog.get('retries', 0)} | "
            f"no-progress {_duration(health.get('remaining'))} remaining"
        ),
    ]
    alerts = snapshot.get("alerts") or []
    for alert in alerts[:3]:
        lines.append(f"[{alert.get('level')}] {alert.get('text')}")
    return "\n".join(lines)


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _duration(seconds) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, round(float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def write_digest(feed: RunFeed) -> dict:
    snapshot = feed.snapshot()
    seed = (snapshot.get("run") or {}).get("seed")
    out_dir = MONITOR_ROOT / (f"seed-{seed}" if seed is not None else "latest")
    _atomic_write(out_dir / "digest.json", json.dumps(snapshot, ensure_ascii=False, indent=2))
    _atomic_write(out_dir / "digest.txt", digest_text(snapshot) + "\n")
    return snapshot


def digest_loop(feed: RunFeed, interval: float, stop: threading.Event) -> None:
    while not stop.wait(interval):
        try:
            write_digest(feed)
        except Exception as exc:  # noqa: BLE001 - a digest failure must not stop the monitor
            print(f"[monitor] digest failed: {type(exc).__name__}: {exc}", flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "S4Monitor/1.0"
    feed: RunFeed

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            try:
                body = INDEX.read_bytes()
            except OSError as exc:
                self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/api/snapshot":
            try:
                payload = self.feed.snapshot()
            except Exception as exc:  # noqa: BLE001 - surface read failures to the browser
                body = json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
                ).encode("utf-8")
                self._send(500, body, "application/json; charset=utf-8")
                return
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path == "/api/digest":
            try:
                text = digest_text(self.feed.snapshot())
            except Exception as exc:  # noqa: BLE001 - digest endpoint reports instead of failing
                text = f"monitor error: {type(exc).__name__}: {exc}"
            self._send(200, text.encode("utf-8"), "text/plain; charset=utf-8")
            return
        if path == "/api/health":
            self._send(200, b'{"ok":true}', "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def log_message(self, *_args) -> None:
        return


def _resolve_run(args) -> Path | None:
    if args.run_dir is not None:
        return Path(args.run_dir).resolve()
    if args.seed is not None:
        return RUNS_ROOT / f"seed-{args.seed}"
    return latest_run_dir()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Read-only live monitor for S4 formal training")
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026))
    parser.add_argument("--run-dir", type=Path, help="Explicit run directory (overrides --seed)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=2.0, help="GPU sample interval seconds")
    parser.add_argument("--digest", action="store_true", help="Print one digest and exit")
    parser.add_argument("--json", action="store_true", help="Print the digest as JSON")
    parser.add_argument("--digest-interval", type=float, default=300.0)
    parser.add_argument("--no-gpu", action="store_true", help="Disable nvidia-smi sampling")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    follow = args.seed is None and args.run_dir is None
    run_dir = _resolve_run(args)
    gpu = None if args.no_gpu else GpuSampler(args.interval)
    if gpu is not None and args.digest:
        gpu.sample_once()
    feed = RunFeed(run_dir, gpu=gpu, follow=follow)
    if args.digest:
        snapshot = feed.snapshot()
        print(
            json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)
            if args.json
            else digest_text(snapshot)
        )
        return 0
    if gpu is not None:
        gpu.start()
    try:
        write_digest(feed)
    except Exception as exc:  # noqa: BLE001 - the server must start even if evidence is unreadable
        print(f"[monitor] initial digest failed: {exc}", flush=True)
    stop = threading.Event()
    threading.Thread(
        target=digest_loop, args=(feed, args.digest_interval, stop), daemon=True
    ).start()
    Handler.feed = feed
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[monitor] serving {url}", flush=True)
    print(f"[monitor] run directory: {feed.run_dir}", flush=True)
    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if gpu is not None:
            gpu.stop()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
