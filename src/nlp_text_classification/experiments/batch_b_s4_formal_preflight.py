"""Bounded CUDA acceptance through the formal S4 training components."""

import os
import time

import numpy as np

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.data.transformer_inputs import EncodedPartition
from nlp_text_classification.experiments.batch_b_s4_formal_state import (
    checked_checkpoint,
    epoch_order,
    latest_checkpoint,
    new_state,
    optimizer_progress,
    restore_checkpoint,
    sampler_cursor,
    save_checkpoint,
    update_rows,
)
from nlp_text_classification.experiments.batch_b_s4_runner import (
    MIN_FREE_VRAM_MIB,
    _rng_probe,
    _state_evidence,
)
from nlp_text_classification.experiments.suite import read_json
from nlp_text_classification.training.progress import write_atomic_json
from nlp_text_classification.training.qwen_s3 import restore_rng
from nlp_text_classification.training.transformer_runner import (
    DevicePartition,
    verify_prediction_file,
)

SEED = 42
SAVED_STEP = 4
NEXT_BATCH = 2


def _identity(binding):
    return binding | {"purpose": "bounded_preflight", "seed": SEED}


def _cursor(engine):
    return sampler_cursor(SEED, 0, NEXT_BATCH, len(engine.data.train), 16)


def _next_update(engine):
    rng = _rng_probe()
    order = epoch_order(SEED, 0, len(engine.data.train))
    outcome = engine.update(update_rows(order, NEXT_BATCH, 16), detail=True)
    return {"rng": rng, "outcome": outcome,
            "updates": optimizer_progress(engine.optimizer, engine.scheduler, SAVED_STEP + 1),
            "parameters": _state_evidence(engine.rank, engine.head, engine.optimizer)}


def run(root, recipe, prepared, binding, gate, progress, engine_factory):
    if latest_checkpoint(root) or (root / "reference.json").exists():
        raise FileExistsError("Choose a new preflight attempt")
    identity = _identity(binding)
    engine = engine_factory(recipe, prepared, SEED, gate, progress)
    initial = _state_evidence(engine.rank, engine.head, engine.optimizer)
    order = epoch_order(SEED, 0, len(engine.data.train))
    normal = update_rows(order, 0, 16)
    warm = [engine.update(update_rows(order, i, 16), detail=True) for i in range(NEXT_BATCH)]
    optimizer_progress(engine.optimizer, engine.scheduler, 2)
    shape = (16, 64)
    encoded = EncodedPartition(
        {"input_ids": np.arange(16 * 64, dtype=np.int64).reshape(shape) + 512,
         "attention_mask": np.ones(shape, dtype=np.int64)},
        np.full(16, 64, dtype=np.int32), engine.encoded["train"].targets[normal],
    )
    partition = DevicePartition(encoded, "cuda")
    progress.emit("STRESS", "16 x 64 forward, backward and AdamW with state resident")
    start = gate.begin_measurement("16x64 start")
    before = time.perf_counter()
    stress = engine.update(np.arange(16), partition=partition, detail=True)
    stress.update(memory=gate.end_measurement("16x64", start),
                  seconds=time.perf_counter() - before)
    if (stress["valid_tokens"], stress["padded_tokens"], stress["gradient_checkpointing"]) != (
        1024, 1024, "full",
    ):
        raise ValueError("Stress test did not exercise all 16 x 64 tokens")
    counts = {name: item["nonzero_tensors"] for name, item in stress["grad_detail"].items()}
    if counts != {"lora_a": 248, "lora_b": 248, "head": 1}:
        raise ValueError(f"Incomplete stress-test gradient coverage: {counts}")
    optimizer_progress(engine.optimizer, engine.scheduler, 3)
    del partition

    tail_indices = update_rows(order, engine.plan["updates_per_epoch"] - 1, 16)
    if len(tail_indices) != 11:
        raise ValueError("Expected 11 rows in the real epoch tail")
    tail = engine.update(tail_indices, detail=True)
    resident = optimizer_progress(engine.optimizer, engine.scheduler, SAVED_STEP)
    changed = _state_evidence(engine.rank, engine.head, engine.optimizer)
    if any(initial[key] == changed[key] for key in initial):
        raise ValueError("Adapter, head or optimizer did not update")
    progress.emit("TAIL_OK", "11 real rows advanced AdamW and the scheduler once", **resident)
    validation = engine.validate()
    if optimizer_progress(engine.optimizer, engine.scheduler, SAVED_STEP) != resident:
        raise ValueError("Validation changed the training step")
    state = new_state() | {"global_step": SAVED_STEP}
    before = time.perf_counter()
    checkpoint = save_checkpoint(root, engine, state, _cursor(engine), identity, validation)
    save_seconds = time.perf_counter() - before
    gate.check("preflight checkpoint")
    progress.emit("CHECKPOINT", str(checkpoint), seconds=save_seconds, **resident)
    next_update = _next_update(engine)
    write_atomic_json(root / "reference.json", {
        "identity": identity, "pid": os.getpid(), "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
        "warm_start_updates": warm, "stress_16x64": stress, "tail_update": tail | {"rows": 11},
        "validation": {key: validation[key] for key in (
            "cutoff", "metrics", "memory", "duration_seconds", "timing",
        )},
        "save_seconds": save_seconds, "cursor": _cursor(engine), "updates": resident,
        "next_update": next_update, "device_min_free_mib": gate.min_free_mib,
    })
    progress.emit("REFERENCE_READY", "Five updates complete, starting independent replay next")


def verify(root, recipe, prepared, binding, gate, progress, engine_factory):
    if (root / "acceptance.json").exists():
        raise FileExistsError("This attempt already has an acceptance receipt")
    reference_file = root / "reference.json"
    reference = read_json(reference_file)
    identity = _identity(binding)
    checkpoint = latest_checkpoint(root)
    if reference["identity"] != identity or reference["pid"] == os.getpid() or (
        checkpoint is None or checkpoint.name != reference["checkpoint"]
        or sha256_file(checkpoint / "checkpoint.json") != reference["checkpoint_sha256"]
    ):
        raise ValueError("Replay requires the exact checkpoint in an independent process")
    receipt = checked_checkpoint(checkpoint, identity)
    engine = engine_factory(recipe, prepared, SEED, gate, progress, saved=checkpoint / "adapter")
    payload = restore_checkpoint(checkpoint, engine, receipt)
    if payload["cursor"] != _cursor(engine) or payload["cursor"] != reference["cursor"]:
        raise ValueError("Next real training batch differs after restoration")
    gate.check("AdamW state restored")
    validation = engine.validate()
    replay, scores = verify_prediction_file(
        checkpoint / "predictions_validation.jsonl", validation["rows"], validation["targets"],
        False, validation["cutoff"], auc=True,
    )
    if not np.array_equal(scores, validation["scores"]) or replay != validation["metrics"] or (
        read_json(checkpoint / "thresholds.json") != validation["thresholds"]
    ):
        raise ValueError("Reloaded canonical validation scores, thresholds or metrics differ")
    restore_rng(payload["rng"])
    result = _next_update(engine)
    if result != reference["next_update"]:
        differences = [key for key in result if result[key] != reference["next_update"][key]]
        raise ValueError(f"Next optimizer update diverged after reload: {differences}")
    gate.check("independent replay completed")
    acceptance = {
        "status": "PASS", "formal_training_started": False, "binding": binding,
        "reference_sha256": sha256_file(reference_file), "checkpoint": checkpoint.name,
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "processes": {"reference": reference["pid"], "verify": os.getpid()},
        "minimum_required_free_mib": MIN_FREE_VRAM_MIB,
        "device_min_free_mib": min(reference["device_min_free_mib"], gate.min_free_mib),
        "stress_16x64": reference["stress_16x64"], "tail_update": reference["tail_update"],
        "validation": reference["validation"] | {"reload_seconds": validation["duration_seconds"],
                                                 "reload_memory": validation["memory"]},
        "save_seconds": reference["save_seconds"],
        "resume_checkpoint_bytes": (checkpoint / "resume.pt").stat().st_size,
        "adapter_bytes": (checkpoint / "adapter" / "adapter_model.safetensors").stat().st_size,
        "restore": {"same_sample_cursor": True, "exact_rng_probe": True,
                    "exact_full_validation_scores": True, "max_abs_score_error": 0.0,
                    "exact_next_update": True, "checkpoint_updates": receipt["updates"],
                    "replayed_updates": result["updates"]},
    }
    write_atomic_json(root / "acceptance.json", acceptance)
    progress.emit("PASS", "Full validation and next optimizer update match across processes",
                  min_free_mib=acceptance["device_min_free_mib"],
                  reproduced_scores=int(scores.size), formal_training_started=False)
