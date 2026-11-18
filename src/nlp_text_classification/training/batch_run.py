"""Record one immutable Batch A attempt with full-precision validation evidence."""

from __future__ import annotations

import csv
import json
import os
import platform
import shlex
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import sklearn
import yaml

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.evaluation.batch_metrics import (
    measure,
    read_predictions,
    select_threshold,
    write_predictions,
)
from nlp_text_classification.training.run import git_commit, git_dirty, write_json


class PrunedRun(Exception):
    """A valid but intentionally stopped Optuna attempt."""


class Tee:
    def __init__(self, console, file):
        self.console = console
        self.file = file

    def write(self, message):
        self.file.write(message)
        return self.console.write(message)

    def flush(self):
        self.file.flush()
        self.console.flush()


class Attempt:
    def __init__(self, output, data, resolved, cohort, seed, progress, *,
                 previous=None, provenance=None):
        dataset, experiment = data.manifest["dataset"], resolved["experiment"]["id"]
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:10]
        self.path = output / dataset / experiment / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "checkpoint").mkdir()
        self.progress = progress
        self.identity = {
            "run_id": run_id, "dataset": dataset, "experiment": experiment,
            "cohort": cohort, "seed": seed, "code_sha": git_commit(), "previous": previous,
            "code_dirty": git_dirty(), "platform": platform.platform(),
            "process_id": os.getpid(),
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "training_manifest_sha256": data.settings["training_manifest_sha256"],
            "command": shlex.join(sys.orig_argv),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if provenance is not None:
            self.identity["transformer"] = provenance
        config_path = self.path / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
        self.identity["config_sha256"] = sha256_file(config_path)
        write_json(self.path / "run_manifest.json", self.identity)
        self.started = time.perf_counter()

    def event(self, phase: str, message: str, **details):
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": phase, "message": message, **details}) + "\n")
        who = f"{self.identity['dataset']}/{self.identity['experiment']}"
        self.progress.emit(phase, f"{who} {message}", **self.identity, **details,
                           stdout_log=self.path / "stdout.log")

    def epoch(self, **values):
        path = self.path / "history.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values))
            if new:
                writer.writeheader()
            writer.writerow(values)
        epoch = values["epoch"]
        macro_f1 = values["val_macro_f1"]
        self.event("EPOCH", f"epoch={epoch} val_macro_f1={macro_f1:.5f}", **values)

    def finish(self, status, message, **details):
        elapsed = time.perf_counter() - self.started
        payload = {"status": status, "reason": message, "elapsed_seconds": elapsed, **details}
        write_json(self.path / "result.json", payload)
        self.event(status, message, elapsed_seconds=elapsed)


def resolve(config, experiment, parameters=None):
    row = next(item for item in config["experiments"] if item["id"] == experiment)
    settings = {**config["svm" if experiment == "E1" else "neural"], **(parameters or {})}
    return {"suite_revision": config["revision"], "experiment": row, "settings": settings,
            "selection": config["selection"], "quality": config["quality"]}


def run_attempt(data, resolved, output: Path, progress, *, cohort: str, seed: int,
                device: str, word2vec=None, pruning=None, previous=None, resume=None,
                on_start=None) -> Path:
    attempt = Attempt(output, data, resolved, cohort, seed, progress, previous=previous)
    if on_start is not None:
        on_start(attempt.path)
    with (attempt.path / "stdout.log").open("x", encoding="utf-8") as out, (
        attempt.path / "stderr.log"
    ).open("x", encoding="utf-8") as err, redirect_stdout(Tee(sys.stdout, out)), (
        redirect_stderr(Tee(sys.stderr, err))
    ):
        try:
            attempt.event("START", f"{cohort} seed={seed} device={device}", device=device)
            if resolved["experiment"]["id"] == "E1":
                from nlp_text_classification.experiments.batch_svm import fit
                scores, reloaded, train_scores, details = fit(data, resolved, attempt, seed)
                kind = "decision"
            else:
                from nlp_text_classification.training.neural_runner import fit
                scores, reloaded, train_scores, details = fit(
                    data, resolved, attempt, seed, device, word2vec, pruning, resume
                )
                kind = "sigmoid"
            y_val = data.encode(data.validation)
            tuning = select_threshold(y_val, scores, data.binary, resolved["selection"], kind)
            cutoff = tuning["threshold"]
            metrics = measure(y_val, scores, cutoff, data.binary)
            quality = resolved["quality"]
            if not np.allclose(scores, reloaded, rtol=quality["reload_score_rtol"],
                               atol=quality["reload_score_atol"]):
                raise ValueError("Reloaded scores differ from the saved model scores")
            if not np.array_equal(scores >= cutoff, reloaded >= cutoff):
                raise ValueError("Checkpoint reload changed discrete validation predictions")
            prediction_path = attempt.path / "predictions_validation.jsonl"
            write_predictions(prediction_path, data.validation, y_val, scores, cutoff)
            _, targets, raw_scores, predictions = read_predictions(prediction_path, score_kind=kind)
            recomputed = measure(targets, raw_scores, cutoff, data.binary)
            if recomputed != metrics or not np.array_equal(predictions, raw_scores >= cutoff):
                raise ValueError("Full-precision prediction evidence did not reproduce metrics")
            write_json(attempt.path / "thresholds.json", tuning)
            write_json(attempt.path / "metrics_validation.json", metrics)
            train_metrics = measure(data.encode(data.train), train_scores, cutoff,
                                    data.binary, auc=False)
            write_json(attempt.path / "metrics_train.json", train_metrics)
            write_json(attempt.path / "quality.json", {
                "reload_scores": True, "reload_predictions": True, "validation_recomputed": True,
                "train_validation_hashes": True, "details": details,
            })
            attempt.finish("PASS", f"val_macro_f1={metrics['macro_f1']:.5f}",
                           macro_f1=metrics["macro_f1"], **details)
        except PrunedRun as exc:
            attempt.finish("PRUNED", str(exc))
        except KeyboardInterrupt:
            attempt.finish("INCOMPLETE", "Interrupted")
            raise
        except Exception as exc:  # noqa: BLE001 - preserve every terminal attempt
            traceback.print_exc()
            attempt.finish("FAIL", str(exc), error_type=type(exc).__name__)
    return attempt.path
