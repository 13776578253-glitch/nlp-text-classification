"""Freeze the S4 Qwen formal runs from validation evidence only.

This module never opens the test partition. It verifies that each run's best
checkpoint still matches its receipt, replays the frozen validation predictions,
independently recomputes the 19-point threshold grid and every reported metric,
and then publishes a content-addressed freeze manifest that a later, separate
entry point can use to read the test split exactly once.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.evaluation.batch_metrics import measure, read_predictions
from nlp_text_classification.experiments.suite import (
    checked_file,
    load_training_data,
    load_yaml,
    read_json,
)
from nlp_text_classification.training.transformer_runner import threshold_record

ROOT = Path("artifacts/batch-b-s4/formal")
OUTPUT = Path("reports/batch-b-s4/formal")
CANDIDATE = Path("reports/batch-b-s4/performance/recommended-training.json")
CANDIDATE_SHA256 = "7d43d2d62bf9a69bc4474198dc189f14da2f74bfda10b22caae9fdd86a19be66"
REFERENCE = OUTPUT / "freeze-ref.json"
SEEDS = (42, 123, 2026)
DATASET_CONFIG = Path("configs/datasets/goemotions.yaml")
SELECTED_LENGTH = 64

BOUND_SOURCES = {
    "batch_b_s4_formal.py": "src/nlp_text_classification/experiments/batch_b_s4_formal.py",
    "batch_b_s4_formal_state.py": "src/nlp_text_classification/experiments/batch_b_s4_formal_state.py",
    "batch_b_s4_formal_preflight.py":
        "src/nlp_text_classification/experiments/batch_b_s4_formal_preflight.py",
    "batch_b_s4_formal_watchdog.py":
        "src/nlp_text_classification/experiments/batch_b_s4_formal_watchdog.py",
    "batch_b_s4_runner.py": "src/nlp_text_classification/experiments/batch_b_s4_runner.py",
    "batch_b_s4_telemetry.py": "src/nlp_text_classification/experiments/batch_b_s4_telemetry.py",
    "batch_b_s3.py": "src/nlp_text_classification/experiments/batch_b_s3.py",
    "qwen_s3.py": "src/nlp_text_classification/training/qwen_s3.py",
    "qwen_s3_streaming.py": "src/nlp_text_classification/training/qwen_s3_streaming.py",
    "transformer_runner.py": "src/nlp_text_classification/training/transformer_runner.py",
    "qwen_s3_inputs.py": "src/nlp_text_classification/data/qwen_s3_inputs.py",
    "batch_metrics.py": "src/nlp_text_classification/evaluation/batch_metrics.py",
    "batch_b_s4_wsl_formal.sh": "scripts/batch_b_s4_wsl_formal.sh",
}
FREEZE_SOURCES = {
    "batch_b_s4_freeze.py": "src/nlp_text_classification/experiments/batch_b_s4_freeze.py",
}


def _bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _publish_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Published bytes differ: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=path.parent) as directory:
        pending = Path(directory) / path.name
        pending.write_bytes(content)
        pending.rename(path)


def _run_dir(seed: int) -> Path:
    return ROOT / "runs" / f"seed-{seed}"


def _completion(run: Path, seed: int) -> dict:
    events = [json.loads(line) for line in
              (run / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    complete = [event for event in events if event.get("phase") == "COMPLETE"]
    if len(complete) != 1:
        raise ValueError(f"seed {seed}: expected exactly one COMPLETE event")
    done = complete[0]
    if done.get("message") not in {"early_stopping", "epoch_cap"}:
        raise ValueError(f"seed {seed}: unexpected stop reason {done.get('message')}")
    epochs = [event for event in events if event.get("phase") == "EPOCH"]
    return {"stop_reason": done.get("message"),
            "epochs": len(epochs), "best": done.get("best"),
            "epoch_records": [{key: row[key] for key in
                               ("epoch", "global_step", "macro_f1", "threshold", "train_loss",
                                "validation_seconds", "elapsed_seconds", "stale_epochs")}
                              for row in epochs]}


def _checkpoint(run: Path, step: int) -> tuple[Path, dict]:
    path = run / "checkpoints" / f"step-{step:08d}"
    receipt = read_json(path / "checkpoint.json")
    if receipt["state"]["global_step"] != step:
        raise ValueError(f"Checkpoint step differs: {path}")
    for name, digest in receipt["files"].items():
        checked_file(path / name, digest)
    if receipt["updates"]["parameter_states"] != 497:
        raise ValueError(f"Trainable parameter state count changed: {path}")
    return path, receipt


def _replay(checkpoint: Path, data, entry: dict, seed: int) -> dict:
    targets = data.encode(data.validation)
    ids, archived_targets, scores, predicted = read_predictions(
        checkpoint / "predictions_validation.jsonl", score_kind="sigmoid")
    if ids != [row.sample_id for row in data.validation] or not np.array_equal(
        archived_targets, targets
    ):
        raise ValueError(f"seed {seed}: validation IDs or targets changed")
    if not np.array_equal(predicted, scores >= entry["threshold"]):
        raise ValueError(f"seed {seed}: archived predictions do not match the frozen threshold")

    thresholds = threshold_record(targets, scores, False, entry["selection"])
    archived_thresholds = read_json(checkpoint / "thresholds.json")
    if thresholds != archived_thresholds:
        raise ValueError(f"seed {seed}: threshold grid replay differs")
    if (thresholds["threshold"], thresholds["macro_f1"]) != (
        entry["threshold"], entry["validation_macro_f1"]
    ):
        raise ValueError(f"seed {seed}: selected threshold or score differs from the record")

    metrics = measure(targets, scores, entry["threshold"], False)
    if metrics != read_json(checkpoint / "metrics_validation.json"):
        raise ValueError(f"seed {seed}: validation metrics replay differs")

    from sklearn.metrics import f1_score

    for candidate in thresholds["candidates"]:
        score = f1_score(targets, scores >= candidate["threshold"], average="macro",
                         zero_division=0)
        if abs(score - candidate["macro_f1"]) > 1e-12:
            raise ValueError(f"seed {seed}: independent threshold F1 differs")

    rows = [json.loads(line) for line in
            (checkpoint / "predictions_validation.jsonl").read_text(encoding="utf-8").splitlines()]
    for row, sample in zip(rows, data.validation, strict=True):
        if row["raw_labels"] != list(sample.labels):
            raise ValueError(f"seed {seed}: raw labels differ from the frozen partition")

    support = targets.sum(axis=0).tolist()
    return {"metrics": metrics, "thresholds": thresholds, "support": support,
            "score_checksum": hashlib.sha256(
                np.ascontiguousarray(scores, dtype=np.float32).tobytes()).hexdigest()}


def freeze() -> dict:
    recipe = read_json(checked_file(CANDIDATE, CANDIDATE_SHA256))
    settings = load_yaml(DATASET_CONFIG)
    data = load_training_data(settings)
    if len(data.train) != 43211 or len(data.validation) != 5416:
        raise ValueError("Frozen partition sizes changed")
    if data.manifest["label_names"] != recipe["label_order"]:
        raise ValueError("Label order differs from the approved recipe")
    selection = recipe["protocol"]["selection"]

    source_hashes = {name: sha256_file(Path(path)) for name, path in
                     {**BOUND_SOURCES, **FREEZE_SOURCES}.items()}
    dataset = {
        "root": settings["frozen_root"],
        "dataset_manifest_sha256": settings["dataset_manifest_sha256"],
        "training_manifest_sha256": settings["training_manifest_sha256"],
        "dataset_manifest": read_json(Path(settings["frozen_root"]) / "dataset_manifest.json"),
        "label_order": data.manifest["label_names"],
        "train_rows": len(data.train),
        "validation_rows": len(data.validation),
        "train_support": data.encode(data.train).sum(axis=0).tolist(),
        "validation_support": data.encode(data.validation).sum(axis=0).tolist(),
    }
    if set(dataset["dataset_manifest"]["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Dataset manifest partitions changed")
    test_record = dataset["dataset_manifest"]["splits"]["test"]

    runs, audits = [], {}
    runtime = accelerator = allocator = None
    for seed in SEEDS:
        run = _run_dir(seed)
        identity = read_json(run / "run.json")["identity"]
        observed = (identity["runtime"], identity["accelerator_packages"],
                    identity["cuda_allocator_settings"])
        if runtime is None:
            runtime, accelerator, allocator = observed
        elif observed != (runtime, accelerator, allocator):
            raise ValueError(f"seed {seed}: runtime differs from the other formal runs")
        if identity.get("seed") != seed or identity.get("purpose") != "formal_train_validation":
            raise ValueError(f"seed {seed}: run identity differs")
        if identity["source_sha256"] != {name: source_hashes[name] for name in BOUND_SOURCES}:
            raise ValueError(f"seed {seed}: bound source fingerprint changed")
        if identity["training_manifest_sha256"] != settings["training_manifest_sha256"]:
            raise ValueError(f"seed {seed}: training manifest differs from the dataset config")

        completion = _completion(run, seed)
        best = completion["best"]
        if best is None:
            raise ValueError(f"seed {seed}: no selected epoch")
        checkpoint, receipt = _checkpoint(run, best["global_step"])
        state = receipt["state"]
        if (state["best"], state["stop_reason"]) != (best, state["stop_reason"]):
            raise ValueError(f"seed {seed}: best epoch record differs from the checkpoint state")
        if best["macro_f1"] != max(entry["macro_f1"] for entry in completion["epoch_records"]):
            raise ValueError(f"seed {seed}: selected epoch is not the validation maximum")

        entry = {"threshold": best["threshold"], "validation_macro_f1": best["macro_f1"],
                 "selection": selection}
        replay = _replay(checkpoint, data, entry, seed)

        runs.append({
            "seed": seed, "run": run.as_posix(), "role": "formal",
            "checkpoint": f"checkpoints/{checkpoint.name}",
            "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "adapter_sha256": receipt["files"]["adapter/adapter_model.safetensors"],
            "resume_sha256": receipt["files"]["resume.pt"],
            "copies": {"predictions_validation.jsonl":
                       receipt["files"]["predictions_validation.jsonl"],
                       "metrics_validation.json": receipt["files"]["metrics_validation.json"],
                       "thresholds.json": receipt["files"]["thresholds.json"]},
            "best_epoch": best["epoch"], "global_step": best["global_step"],
            "train_loss": best["train_loss"],
            "threshold": best["threshold"], "score_kind": "sigmoid",
            "validation_macro_f1": best["macro_f1"],
            "stop_reason": completion["stop_reason"], "epochs_run": completion["epochs"],
            "optimizer_step": receipt["updates"]["optimizer_step"],
            "parameter_states": receipt["updates"]["parameter_states"],
        })
        audits[f"seed-{seed}"] = {
            "status": "PASS", "validation_rows": len(data.validation),
            "threshold_candidates": len(replay["thresholds"]["candidates"]),
            "threshold_replay_exact": True, "metrics_replay_exact": True,
            "score_checksum": replay["score_checksum"],
            "checkpoint_files": len(receipt["files"]),
            "macro_f1": replay["metrics"]["macro_f1"],
            "micro_f1": replay["metrics"]["micro_f1"],
            "pr_auc": replay["metrics"]["pr_auc"],
            "hamming_loss": replay["metrics"]["hamming_loss"],
            "epoch_records": completion["epoch_records"],
        }

    frozen = {
        "status": "frozen",
        "scope": "validation_only",
        "test_read": False,
        "dataset": dataset,
        "test_partition": test_record,
        "label_order": data.manifest["label_names"],
        "recipe": {"path": CANDIDATE.as_posix(), "sha256": CANDIDATE_SHA256,
                   "candidate_id": recipe["s4_candidate_id"],
                   "model_id": recipe["protocol"]["model_id"],
                   "model_revision": recipe["protocol"]["model_revision"],
                   "max_length": recipe["max_length"],
                   "selected_length": SELECTED_LENGTH,
                   "quantization": recipe["protocol"]["quantization"],
                   "lora": recipe["protocol"]["lora"],
                   "training": recipe["protocol"]["training"],
                   "selection": selection,
                   "plan": recipe["protocol"].get("plan")},
        "selection_rule": {
            "metric": selection["metric"],
            "tie_atol": selection["score_tie_atol"],
            "threshold_grid": selection["threshold_grids"]["sigmoid"],
            "threshold_application": "one global sigmoid threshold for all 28 labels; score >= threshold",
            "epoch_rule": "maximum validation macro-F1; ties keep the earlier epoch",
            "early_stopping": {"patience": recipe["protocol"]["training"]["patience"],
                               "max_epochs": recipe["protocol"]["training"]["epochs"]},
            "aggregation": "per-seed selected validation score; mean and sample std (ddof=1) across the three seeds",
        },
        "runs": runs,
        "source_sha256": source_hashes,
        "environment": runtime,
        "accelerator_packages": accelerator,
        "cuda_allocator_settings": allocator,
        "preflight": {"attempt": "prelaunch-v6",
                      "acceptance_sha256": sha256_file(OUTPUT / "acceptance.json")},
    }
    content = _bytes(frozen)
    digest = hashlib.sha256(content).hexdigest()
    path = OUTPUT / "frozen" / f"{digest}.json"
    _publish_once(path, content)
    _publish_once(OUTPUT / "freeze-audit.json", _bytes({
        "status": "PASS", "scope": "validation_only", "test_read": False,
        "freeze_sha256": digest, "runs": audits,
        "summary": _summary(audits),
    }))
    _publish_once(REFERENCE, _bytes({"freeze": path.as_posix(), "sha256": digest}))
    return {"freeze_sha256": digest, "freeze": path.as_posix(), "runs": len(runs)}


def _summary(audits: dict) -> dict:
    macro = [item["macro_f1"] for item in audits.values()]
    micro = [item["micro_f1"] for item in audits.values()]
    auc = [item["pr_auc"] for item in audits.values()]
    hamming = [item["hamming_loss"] for item in audits.values()]
    mean = float(np.mean(macro))
    std = float(np.std(macro, ddof=1)) if len(macro) > 1 else None
    return {"seeds": sorted(item.split("-")[1] for item in audits),
            "macro_f1_per_seed": macro, "macro_f1_mean": mean, "macro_f1_std": std,
            "micro_f1_per_seed": micro, "micro_f1_mean": float(np.mean(micro)),
            "pr_auc_per_seed": auc, "pr_auc_mean": float(np.mean(auc)),
            "hamming_per_seed": hamming, "hamming_mean": float(np.mean(hamming))}


def verified_freeze() -> tuple[dict, str]:
    reference = read_json(REFERENCE)
    digest = reference["sha256"]
    path = OUTPUT / "frozen" / f"{digest}.json"
    if reference["freeze"] != path.as_posix():
        raise ValueError("Unexpected freeze reference")
    frozen = read_json(checked_file(path, digest))
    if (frozen["status"], frozen["scope"], frozen["test_read"]) != ("frozen", "validation_only",
                                                                      False):
        raise ValueError("Freeze is not a validation-only frozen manifest")
    if sha256_file(CANDIDATE) != frozen["recipe"]["sha256"]:
        raise ValueError("Approved recipe changed after freezing")
    for name, source in {**BOUND_SOURCES, **FREEZE_SOURCES}.items():
        if sha256_file(Path(source)) != frozen["source_sha256"][name]:
            raise ValueError(f"Freeze source changed: {name}")
    for entry in frozen["runs"]:
        run = Path(entry["run"])
        checkpoint, _ = _checkpoint(run, entry["global_step"])
        if (entry["checkpoint"], entry["checkpoint_sha256"]) != (
            f"checkpoints/{checkpoint.name}", sha256_file(checkpoint / "checkpoint.json")
        ):
            raise ValueError(f"{entry['run']}: checkpoint receipt changed")
        if entry["threshold"] != read_json(checkpoint / "thresholds.json")["threshold"]:
            raise ValueError(f"{entry['run']}: frozen threshold differs from the checkpoint")
    audit = read_json(OUTPUT / "freeze-audit.json")
    if audit["status"] != "PASS" or audit["freeze_sha256"] != digest:
        raise ValueError("Freeze audit does not match the manifest")
    return frozen, digest
