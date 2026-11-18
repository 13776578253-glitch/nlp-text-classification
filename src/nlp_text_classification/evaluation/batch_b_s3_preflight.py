"""Verify S3 preflight receipts and resolve the next-stage training configuration."""

import json
from pathlib import Path

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.experiments.batch_b_s3_resources import MANIFEST, ROOT, SUITE
from nlp_text_classification.experiments.suite import checked_file, load_yaml, read_json
from nlp_text_classification.training.progress import write_atomic_json


def resolve(attempt: Path, suite):
    receipts = {name: read_json(attempt / f"{name}.json") for name in (
        "load-audit", "lora-audit", "tokenization", "functional", "benchmark", "recovery",
        "environment", "kernel",
    )}
    load, lora, tokens, functional, bench, recovery, environment, kernel = receipts.values()
    for name in ("load-audit", "lora-audit", "benchmark", "recovery", "kernel"):
        if receipts[name]["status"] != "PASS":
            raise ValueError(f"Preflight acceptance has not passed: {name}")
    if load["text_weights"] != 426 or load["missing_keys"] != ["score.weight"] or (
        lora["targets"] != load["quantized"] or len(lora["targets"]) != 248
        or sum(lora["counts"].values()) != 65001472
    ):
        raise ValueError("Classification head or LoRA coverage is incomplete")
    if recovery["original_pid"] == recovery["recovery_pid"] or (
        not recovery["exact_next_update"] or not recovery["exact_rng_probe"]
    ):
        raise ValueError("Independent optimizer and RNG recovery is incomplete")
    if environment != kernel["environment"] or bench["suite_sha256"] != sha256_file(SUITE):
        raise ValueError("Runtime environment or suite differs from preflight")
    if functional["stress"]["sequence_length"] < tokens["selected_length"] or (
        functional["pooling"]["logit_shape"] != [2, 28]
    ):
        raise ValueError("Functional probes do not cover the locked model and maximum length")
    updates = attempt / "updates.jsonl"
    with updates.open(encoding="utf-8") as stream:
        history = [json.loads(line) for line in stream]
    warm, measured = bench["warmup_updates"], bench["stable_updates"]
    if measured < 100 or [row["update"] for row in history] != list(range(1, warm + measured + 1)):
        raise ValueError("Stable optimizer update window is incomplete")
    if abs(sum(row["seconds"] for row in history[warm:]) - bench["stable_seconds"]) > 1e-6:
        raise ValueError("Measured update timing does not match its receipt")
    for file, digest in bench["checkpoint_files"].items():
        checked_file(attempt / file, digest)
    manifest = read_json(MANIFEST)
    if tokens["model_revision"] != manifest["revision"] or (
        tokens["training_manifest_sha256"] != load_yaml(Path(
            suite["dataset_config"]
        ))["training_manifest_sha256"]
    ):
        raise ValueError("Tokenization did not use the locked model and data")
    return {
        "status": "ready_for_review", "scope": "train_and_validation_only",
        "config": suite, "max_length": tokens["selected_length"],
        "label_order": tokens["label_order"], "dataset_manifest_sha256": load_yaml(
            Path(suite["dataset_config"])
        )["dataset_manifest_sha256"],
        "training_manifest_sha256": tokens["training_manifest_sha256"],
        "environment": environment, "model_resources_sha256": sha256_file(MANIFEST),
        "evidence": {path.relative_to(ROOT).as_posix(): sha256_file(path)
                     for path in (updates, *(attempt / f"{name}.json" for name in receipts))},
    }


def deliver(attempt, suite, progress):
    plan = resolve(attempt, suite)
    deliverable = attempt / "resolved-training-config.json"
    write_atomic_json(deliverable, plan)
    progress.emit("READY_FOR_REVIEW", "S3 preflight accepted; formal training awaits review",
                  max_length=plan["max_length"], config_sha256=sha256_file(deliverable))
