"""Tests for the standalone S4 training monitor."""

import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "apps" / "monitor" / "s4_monitor.py"
_SPEC = importlib.util.spec_from_file_location("s4_monitor", MODULE_PATH)
monitor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(monitor)


def iso(stamp):
    return datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_jsonl_tail_handles_partial_lines(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.write_bytes(b'{"a": 1}\n{"b": 2}\n{"c"')
    tail = monitor.JsonlTail(path)
    assert len(tail.poll()) == 2
    assert tail.rows == [{"a": 1}, {"b": 2}]
    with path.open("ab") as stream:
        stream.write(b": 3}\n")
    assert len(tail.poll()) == 1
    assert tail.rows[-1] == {"c": 3}


def test_plan_uses_run_manifest(tmp_path):
    run = tmp_path / "seed-42"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps(
            {
                "identity": {
                    "plan": {
                        "effective_batch_size": 16,
                        "updates_per_epoch": 2701,
                        "total_updates": 13505,
                        "warmup_updates": 811,
                    },
                    "checkpoint_interval": 500,
                },
                "protocol": {"training": {"epochs": 5}},
            }
        ),
        encoding="utf-8",
    )
    plan = monitor.plan_for(run)
    assert plan["total_updates"] == 13505
    assert plan["updates_per_epoch"] == 2701
    assert plan["epochs"] == 5
    assert plan["checkpoint_interval"] == 500


def test_status_stalled_and_recovering(tmp_path):
    now = time.time()
    rows = [{"time": iso(now - 200), "phase": "TRAIN", "global_step": 25}]
    watchdog = [{"time": iso(now - 200), "event": "WORKER_STARTED", "pid": 7}]
    health = monitor._status(rows, watchdog, now, tmp_path)
    assert health["status"] == "STALLED"
    assert health["remaining"] < 0

    recovering = [{"time": iso(now - 5), "event": "RETRY", "mode": "resume"}]
    assert monitor._status(rows, recovering, now, tmp_path)["status"] == "RECOVERING"

    exited = [{"time": iso(now - 5), "event": "WORKER_EXITED", "returncode": 75}]
    assert monitor._status(rows, exited, now, tmp_path)["status"] == "EXITED"


def test_status_uses_new_worker_start_for_startup(tmp_path):
    now = time.time()
    rows = [{"time": iso(now - 900), "phase": "LOAD_MODEL", "pid": 1}]
    watchdog = [{"time": iso(now - 10), "event": "WORKER_STARTED", "pid": 2}]
    health = monitor._status(rows, watchdog, now, tmp_path)
    assert health["status"] == "STARTING"
    assert health["budget"] == 300


def test_eta_uses_measured_epoch_wall():
    plan = {"updates_per_epoch": 2701, "total_updates": 13505, "epochs": 5}
    epochs = [{"elapsed_seconds": 2000.0, "macro_f1": 0.5}]
    eta = monitor.estimate_eta(5402, 0, 3, plan, epochs, 1.3, None)
    assert eta["confidence"] == "epoch"
    assert eta["seconds"] == pytest.approx(2701 / 1.3 + 2 * 2000.0)
    assert monitor.estimate_eta(13505, 2701, 5, plan, epochs, 1.3, "epoch_cap")["seconds"] == 0


def test_snapshot_reads_progress_grad_and_metrics(tmp_path):
    run = tmp_path / "seed-42"
    checkpoint = run / "checkpoints" / "step-00000025"
    checkpoint.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "identity": {
                    "plan": {
                        "effective_batch_size": 16,
                        "updates_per_epoch": 2701,
                        "total_updates": 13505,
                        "warmup_updates": 811,
                    },
                    "checkpoint_interval": 500,
                },
                "protocol": {"training": {"epochs": 5}},
            }
        ),
        encoding="utf-8",
    )
    now = time.time()
    write_jsonl(
        run / "progress.jsonl",
        [
            {"time": iso(now - 30), "phase": "LOAD_MODEL", "pid": 11},
            {
                "time": iso(now - 20),
                "phase": "TRAIN",
                "message": "Epoch 1, update 25/2701",
                "global_step": 25,
                "loss": 1.25,
                "gradient_norm": 0.8,
                "learning_rates": [1e-04, 5e-04],
                "device_min_free_mib": 4096,
                "pid": 11,
            },
            {
                "time": iso(now - 5),
                "phase": "TRAIN",
                "message": "Epoch 1, update 50/2701",
                "global_step": 50,
                "loss": 1.1,
                "gradient_norm": 0.7,
                "learning_rates": [2e-04, 1e-03],
                "valid_tokens": 320,
                "padded_tokens": 512,
                "device_min_free_mib": 4000,
                "pid": 11,
            },
        ],
    )
    write_jsonl(
        run / "watchdog.jsonl",
        [{"time": iso(now - 40), "event": "WORKER_STARTED", "pid": 11}],
    )
    best = {"epoch": 1, "global_step": 25, "macro_f1": 0.51, "threshold": 0.35, "train_loss": 1.3}
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "state": {
                    "global_step": 50,
                    "epoch": 1,
                    "batch": 50,
                    "best": best,
                    "stale_epochs": 0,
                    "stop_reason": None,
                }
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "metrics_validation.json").write_text(
        json.dumps({"macro_f1": 0.51, "micro_f1": 0.52, "hamming_loss": 0.04, "pr_auc": 0.61}),
        encoding="utf-8",
    )
    feed = monitor.RunFeed(run, gpu=None, follow=False)
    snapshot = feed.snapshot()
    assert snapshot["status"] == "RUNNING"
    assert snapshot["run"]["seed"] == 42
    assert snapshot["progress"]["global_step"] == 50
    assert snapshot["progress"]["gradient_norm"] == pytest.approx(0.7)
    assert snapshot["progress"]["vram_train_min_mib"] == 4000
    assert snapshot["progress"]["epoch_fraction"] == pytest.approx(50 / 2701)
    assert snapshot["best"]["macro_f1"] == pytest.approx(0.51)
    assert snapshot["checkpoint"]["step"] == 50
    assert snapshot["metrics"]["micro_f1"] == pytest.approx(0.52)
    assert snapshot["metrics_step"] == 25
    assert len(snapshot["label_order"]) == 28
    assert snapshot["label_order"][0] == "admiration"
    assert snapshot["train_history"][-1]["valid_tokens"] == 320
    assert snapshot["train_history"][-1]["padded_tokens"] == 512
    assert len(snapshot["notable_events"]) == 1
    assert snapshot["notable_events"][0]["phase"] == "LOAD_MODEL"
    assert snapshot["vram"]["error"] == "disabled"
    text = monitor.digest_text(snapshot)
    assert "S4 seed 42" in text
    assert "grad 0.7000" in text
    assert "best 0.5100 @ epoch 1" in text


def test_snapshot_without_run_is_explicit(tmp_path):
    feed = monitor.RunFeed(None, gpu=None, follow=False)
    snapshot = feed.snapshot()
    assert snapshot["status"] == "NO_RUN"
    assert snapshot["run"] is None
    assert "no training run found" in monitor.digest_text(snapshot)


def test_halt_alert_only_when_it_is_current():
    now = time.time()
    halt = {"time": iso(now - 100), "phase": "HALTED", "error": "memory"}
    later = {"time": iso(now - 10), "phase": "TRAIN"}
    alerts = monitor._alerts("RUNNING", later, [], [], [halt], now)
    assert not any("HALTED" in alert["text"] for alert in alerts)
    alerts = monitor._alerts("HALTED", halt, [], [], [halt], now)
    assert any("HALTED" in alert["text"] for alert in alerts)


def test_gpu_view_without_sampler(tmp_path):
    feed = monitor.RunFeed(tmp_path / "seed-42", gpu=None, follow=False)
    view = feed._gpu_view(time.time())
    assert view["samples"] == []
    assert view["error"] == "disabled"
    assert view["warn_mib"] == 2560
