"""Run the approved S4 configuration behind an independent preflight gate."""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.evaluation.batch_metrics import measure
from nlp_text_classification.experiments.batch_b_s3 import configure_device, environment, inputs
from nlp_text_classification.experiments.batch_b_s3_perf import _package_versions
from nlp_text_classification.experiments.batch_b_s3_resources import MANIFEST, SUITE
from nlp_text_classification.experiments.batch_b_s4_formal_state import (
    checked_checkpoint,
    epoch_order,
    latest_checkpoint,
    new_state,
    restore_checkpoint,
    sampler_cursor,
    save_checkpoint,
    select_epoch,
    update_rows,
)
from nlp_text_classification.experiments.batch_b_s4_formal_watchdog import (
    CUDA_OOM_EXIT,
    ROOT,
    VRAM_WARNING_MIB,
    parse_args,
)
from nlp_text_classification.experiments.batch_b_s4_runner import (
    MIN_FREE_VRAM_MIB,
    build_model_s4,
    evaluate_subset,
    make_optimizer,
    run_update,
    update_budget,
)
from nlp_text_classification.experiments.batch_b_s4_telemetry import GpuTelemetryMonitor
from nlp_text_classification.experiments.batch_scheduler import exclusive_queue
from nlp_text_classification.experiments.suite import checked_file, read_json
from nlp_text_classification.training.progress import Progress, write_atomic_json
from nlp_text_classification.training.qwen_s3 import restore_rng, seed_all
from nlp_text_classification.training.resources import GIB, configure_compute_threads, protect
from nlp_text_classification.training.transformer_runner import DevicePartition, threshold_record

CANDIDATE = Path("reports/batch-b-s4/performance/recommended-training.json")
CANDIDATE_SHA256 = "7d43d2d62bf9a69bc4474198dc189f14da2f74bfda10b22caae9fdd86a19be66"
LOCKED_PLAN = {"effective_batch_size": 16, "updates_per_epoch": 2701, "tail_rows": 11,
               "total_updates": 13505, "warmup_updates": 811}
CHECKPOINT_INTERVAL = 500
LONG_BATCH_LENGTH = 48
ALLOCATOR_SETTINGS = "expandable_segments:True"
MIB = 1 << 20


class GpuHeadroom:
    def __init__(self, monitor, progress, *, strict):
        self.monitor = monitor
        self.progress = progress
        self.strict = strict
        self.samples_read = 0
        self.min_free_mib = float("inf")
        self.warning_active = False

    def check(self, phase):
        samples = self.monitor.samples[self.samples_read:]
        self.samples_read += len(samples)
        available = torch.cuda.mem_get_info()[0] // MIB
        recent = min(available, *(item["free_mib"] for item in samples)) if samples else available
        self.min_free_mib = min(self.min_free_mib, recent)
        if self.strict and self.min_free_mib < MIN_FREE_VRAM_MIB:
            raise RuntimeError(f"{phase}: minimum device free VRAM {self.min_free_mib} MiB "
                               f"is below {MIN_FREE_VRAM_MIB} MiB "
                               f"(current CUDA free {available} MiB, peak allocated "
                               f"{torch.cuda.max_memory_allocated() // MIB} MiB, peak reserved "
                               f"{torch.cuda.max_memory_reserved() // MIB} MiB)")
        if not self.strict:
            if not self.warning_active and recent < VRAM_WARNING_MIB:
                self.progress.emit("VRAM_WARNING", phase, recent_free_mib=recent,
                                   warning_mib=VRAM_WARNING_MIB)
                self.warning_active = True
            elif self.warning_active and recent >= VRAM_WARNING_MIB + 256:
                self.warning_active = False
        return available

    def begin_measurement(self, phase):
        start = self.check(phase)
        reserved = torch.cuda.memory_reserved() // MIB
        torch.cuda.reset_peak_memory_stats()
        return start + reserved

    def end_measurement(self, phase, adjusted_start):
        torch.cuda.synchronize()
        self.check(phase)
        peak = (torch.cuda.max_memory_reserved() + MIB - 1) // MIB
        bound = adjusted_start - peak
        self.min_free_mib = min(self.min_free_mib, bound)
        self.check(phase)
        return {"device_min_free_mib": self.min_free_mib, "phase_allocator_bound_mib": bound,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                "peak_reserved_mib": peak}


def load_recipe():
    recipe = read_json(checked_file(CANDIDATE, CANDIDATE_SHA256))
    checked_file(SUITE, recipe["training_suite_sha256"])
    checked_file(Path(recipe["runtime"]["package_lock"]), recipe["runtime"]["package_lock_sha256"])
    reported = checked_file(Path("reports/batch-b-s3/performance/model-resources.json"),
                            recipe["model_resources_sha256"])
    if read_json(reported) != read_json(MANIFEST):
        raise ValueError("Official model resource manifest differs from the approved baseline")
    return recipe


def _binding(recipe):
    actual = environment()
    actual["capability"] = list(actual["capability"])
    expected = recipe["runtime"]
    if (sys.platform, actual, _package_versions()) != (
        "linux", expected["observed"], expected["accelerator_packages"]
    ) or "microsoft" not in os.uname().release.lower():
        raise ValueError("Formal runs require the approved WSL2, CUDA and accelerator packages")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != ALLOCATOR_SETTINGS:
        raise ValueError("Launch the formal entry through the approved CUDA allocator wrapper")
    source = Path(__file__).parent
    files = [source / name for name in (
        "batch_b_s4_formal.py", "batch_b_s4_formal_state.py", "batch_b_s4_formal_preflight.py",
        "batch_b_s4_formal_watchdog.py", "batch_b_s4_runner.py", "batch_b_s4_telemetry.py",
        "batch_b_s3.py",
    )] + [source.parent / name for name in (
        "training/qwen_s3.py", "training/qwen_s3_streaming.py", "training/transformer_runner.py",
        "data/qwen_s3_inputs.py", "evaluation/batch_metrics.py",
    )] + [Path("scripts/batch_b_s4_wsl_formal.sh")]
    return {
        "candidate_sha256": CANDIDATE_SHA256, "model_revision": recipe["protocol"]["model_revision"],
        "runtime": actual, "accelerator_packages": expected["accelerator_packages"],
        "training_manifest_sha256": recipe["training_manifest_sha256"],
        "encoded_sha256": {name: item["encoded_sha256"]
                           for name, item in recipe["partitions"].items()},
        "source_sha256": {path.name: sha256_file(path) for path in files},
        "plan": LOCKED_PLAN, "checkpoint_interval": CHECKPOINT_INTERVAL,
        "long_batch_memory": {"input_length_above": LONG_BATCH_LENGTH,
                              "release_idle_cache": True, "gradient_checkpointing": "full"},
        "cuda_allocator_settings": ALLOCATOR_SETTINGS,
    }


def checked_inputs(recipe, progress):
    progress.emit("VERIFY_INPUTS", "Checking frozen train/validation and official model files")
    snapshot, data, tokens, encoded = inputs(recipe["protocol"])
    keys = ("selected_length", "label_order", "training_manifest_sha256", "partitions")
    expected = recipe | {"selected_length": recipe["max_length"]}
    if any(tokens[key] != expected[key] for key in keys) or (
        set(encoded) != {"train", "validation"}
    ):
        raise ValueError("Formal inputs differ from the approved tokenization and labels")
    plan = update_budget(recipe["protocol"]["training"], len(data.train))
    if plan != LOCKED_PLAN or len(data.validation) != 5416:
        raise ValueError("Formal optimizer or validation budget changed")
    progress.emit("INPUTS_OK", "Verified 43,211 train and 5,416 validation rows", **plan)
    return snapshot, data, encoded


class Engine:
    def __init__(self, recipe, prepared, seed, gate, progress, *, saved=None):
        snapshot, self.data, self.encoded = prepared
        self.suite, self.gate, self.progress = recipe["protocol"], gate, progress
        self.options = self.suite["training"]
        seed_all(seed)
        progress.emit("LOAD_MODEL", "Loading official NF4 base and classification adapter",
                      seed=seed, source="fresh" if saved is None else "checkpoint")
        self.model, self.rank, self.head, self.lora = build_model_s4(
            snapshot, self.suite, recipe["label_order"], precision_mode="nf4_fp32_stream",
            gc_mode=self.options["gradient_checkpointing"], saved=saved, fresh=saved is None,
        )
        gate.check("model load")
        self.optimizer, self.scheduler, self.plan = make_optimizer(
            self.rank, self.head, self.options, len(self.data.train),
            foreach=self.options["adamw_foreach"],
        )
        self.trainable = [*self.rank.values(), *self.head.values()]
        self.train = DevicePartition(self.encoded["train"], "cuda")
        self.extra_checkpoint_layers = [layer for layer in self.model.get_base_model().model.layers
                                        if not layer.gradient_checkpointing]
        if len(self.extra_checkpoint_layers) != 16:
            raise ValueError("Expected 16 layers outside the normal checkpointing pattern")

    def update(self, indices, *, partition=None, detail=False):
        batch = self.train if partition is None else partition
        long_batch = batch.lengths[indices].max() > LONG_BATCH_LENGTH
        extra_layers = self.extra_checkpoint_layers if long_batch else ()
        if long_batch:
            torch.cuda.empty_cache()
        for layer in extra_layers:
            layer.gradient_checkpointing = True
        try:
            outcome = run_update(
                self.model, self.optimizer, self.scheduler, self.suite,
                batch, indices, self.trainable,
                micro_batch_size=self.options["micro_batch_size"], padding_mode="dynamic_mult8",
                sort_mode="unsorted", check_grad_detail=detail,
            )
            self.gate.check("optimizer update")
        finally:
            for layer in extra_layers:
                layer.gradient_checkpointing = False
        if long_batch:
            torch.cuda.empty_cache()
        outcome["gradient_checkpointing"] = (
            "full" if long_batch else self.options["gradient_checkpointing"]
        )
        return outcome

    def validate(self):
        rows = self.data.validation
        self.progress.emit("VALIDATE", "Scoring the full validation partition", rows=len(rows))
        torch.cuda.empty_cache()
        before = self.gate.begin_measurement("validation start")

        def on_batch(done):
            self.gate.check("validation batch")
            if done == len(rows) or done % 1024 == 0:
                self.progress.emit("VALIDATION_ROWS", f"{done}/{len(rows)} validation rows")

        _, scores, seconds, timing = evaluate_subset(
            self.model, self.encoded["validation"], np.arange(len(rows)),
            batch_size=self.options["evaluation_micro_batch_size"], multiple=8,
            sort_by_length=self.options["evaluation_sort_by_length"], on_batch=on_batch,
        )
        memory = self.gate.end_measurement("validation", before)
        targets = self.data.encode(rows)
        if not np.array_equal(targets, self.encoded["validation"].targets):
            raise ValueError("Canonical validation label order changed")
        thresholds = threshold_record(targets, scores, False, self.suite["selection"])
        cutoff = thresholds["threshold"]
        metrics = measure(targets, scores, cutoff, binary=False)
        for candidate in thresholds["candidates"]:
            score = f1_score(targets, scores >= candidate["threshold"], average="macro",
                             zero_division=0)
            if abs(score - candidate["macro_f1"]) > 1e-12:
                raise ValueError("Independent validation threshold replay differs")
        return {"rows": rows, "scores": scores, "targets": targets, "cutoff": cutoff,
                "metrics": metrics, "thresholds": thresholds, "memory": memory,
                "duration_seconds": seconds, "timing": timing}


def _cursor(engine, seed, state):
    if state["stop_reason"] is not None:
        return None
    return sampler_cursor(seed, state["epoch"], state["batch"], len(engine.data.train),
                          engine.plan["effective_batch_size"])


def _train(root, recipe, prepared, seed, identity, gate, progress, *, resume):
    existing = latest_checkpoint(root)
    manifest = root / "run.json"
    if resume:
        if existing is None or read_json(manifest)["identity"] != identity:
            raise ValueError("Resume requires a complete checkpoint and matching formal Run")
    else:
        if existing is not None:
            raise FileExistsError(f"Use resume for the existing Run at {root}")
        if manifest.exists():
            if read_json(manifest)["identity"] != identity:
                raise ValueError("Incomplete Run was created with a different contract")
        else:
            write_atomic_json(manifest, {"identity": identity, "protocol": recipe["protocol"]})
    receipt = checked_checkpoint(existing, identity) if existing else None
    if receipt and receipt["state"]["stop_reason"]:
        progress.emit("COMPLETE", "Run already completed", best=receipt["state"]["best"])
        return
    engine = Engine(recipe, prepared, seed, gate, progress,
                    saved=existing / "adapter" if existing else None)
    if receipt:
        payload = restore_checkpoint(existing, engine, receipt)
        state = payload["state"]
        if payload["cursor"] != _cursor(engine, seed, state):
            raise ValueError("Saved epoch permutation or next sample cursor changed")
        restore_rng(payload["rng"])
        progress.emit("RESUMED", str(existing), global_step=state["global_step"],
                      epoch=state["epoch"] + 1, batch=state["batch"])
    else:
        state = new_state()
        save_checkpoint(root, engine, state, _cursor(engine, seed, state), identity)

    updates = engine.plan["updates_per_epoch"]
    effective = engine.plan["effective_batch_size"]
    while state["stop_reason"] is None:
        epoch_started = time.perf_counter()
        epoch = state["epoch"]
        order = epoch_order(seed, epoch, len(engine.data.train))
        for batch in range(state["batch"], updates):
            indices = update_rows(order, batch, effective)
            row = engine.update(indices)
            step = state["global_step"] + 1
            state.update(global_step=step, batch=batch + 1,
                         loss_sum=state["loss_sum"] + row["loss"] * len(indices),
                         loss_rows=state["loss_rows"] + len(indices))
            if batch == 0 or (batch + 1) % 25 == 0 or batch + 1 == updates:
                progress.emit("TRAIN", f"Epoch {epoch + 1}, update {batch + 1}/{updates}",
                              global_step=step, loss=row["loss"],
                              gradient_norm=row["gradient_norm"], learning_rates=row["used_lrs"],
                              valid_tokens=row["valid_tokens"], padded_tokens=row["padded_tokens"],
                              gradient_checkpointing=row["gradient_checkpointing"],
                              device_min_free_mib=gate.min_free_mib)
            if step % CHECKPOINT_INTERVAL == 0 and batch + 1 < updates:
                saved = save_checkpoint(root, engine, state, _cursor(engine, seed, state), identity)
                progress.emit("CHECKPOINT", str(saved), global_step=step)
        validation = engine.validate()
        state = select_epoch(state, validation, engine.options, engine.suite["selection"], updates)
        saved = save_checkpoint(root, engine, state, _cursor(engine, seed, state), identity, validation)
        progress.emit("EPOCH", str(saved), **state["epochs"][-1], best=state["best"],
                      validation_seconds=validation["duration_seconds"],
                      elapsed_seconds=time.perf_counter() - epoch_started,
                      stale_epochs=state["stale_epochs"])
    progress.emit("COMPLETE", state["stop_reason"], best=state["best"], epochs=state["epoch"])


def main():
    args = parse_args()
    if args.mode == "status":
        latest = latest_checkpoint(ROOT / "runs" / f"seed-{args.seed}")
        state = read_json(latest / "checkpoint.json")["state"] if latest else None
        print(json.dumps({"checkpoint": str(latest) if latest else None, "state": state}, indent=2))
        return
    preflight = args.mode in {"preflight", "verify-preflight"}
    attempt = args.attempt if preflight else args.preflight
    evidence = ROOT / "preflight" / attempt
    root = evidence if preflight else ROOT / "runs" / f"seed-{args.seed}"
    progress = Progress(root / "progress.md", " ".join(sys.orig_argv), title="Batch B S4 formal")
    progress.context.update(mode=args.mode, pid=os.getpid())
    configure_compute_threads()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        recipe = load_recipe()
        with exclusive_queue(ROOT.parents[1] / ".batch-a-gpu.lock"), protect(
            progress, process_limit=recipe["protocol"]["preflight"]["process_commit_limit_gib"] * GIB,
        ), GpuTelemetryMonitor(interval=0.25) as monitor:
            configure_device()
            binding = _binding(recipe)
            if not preflight:
                acceptance = read_json(evidence / "acceptance.json")
                if acceptance["status"] != "PASS" or acceptance["binding"] != binding:
                    raise ValueError("Formal run requires the matching passed preflight")
                checked_file(evidence / "reference.json", acceptance["reference_sha256"])
                checked_file(evidence / "checkpoints" / acceptance["checkpoint"] / "checkpoint.json",
                             acceptance["checkpoint_sha256"])
            gate = GpuHeadroom(monitor, progress, strict=preflight)
            gate.check("startup")
            prepared = checked_inputs(recipe, progress)
            if preflight:
                from nlp_text_classification.experiments.batch_b_s4_formal_preflight import (
                    run as run_preflight,
                )
                from nlp_text_classification.experiments.batch_b_s4_formal_preflight import verify

                action = run_preflight if args.mode == "preflight" else verify
                action(root, recipe, prepared, binding, gate, progress, Engine)
            else:
                identity = binding | {"purpose": "formal_train_validation", "seed": args.seed}
                _train(root, recipe, prepared, args.seed, identity, gate, progress,
                       resume=args.mode == "resume")
    except (Exception, KeyboardInterrupt) as exc:
        progress.emit("HALTED", str(exc), error=type(exc).__name__, mode=args.mode)
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            traceback.print_exc()
            raise SystemExit(CUDA_OOM_EXIT) from exc
        raise


if __name__ == "__main__":
    main()
