"""Audit, preflight and run the fixed Batch B S1 encoder matrix."""

from __future__ import annotations

import csv
import gc
import math
import shlex
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.data.transformer_inputs import audit_tokenization, digest_json
from nlp_text_classification.experiments.batch_b_resources import (
    OUTPUT,
    RESOURCE_LOCK,
    SUITE,
    label_names,
    load_config,
    load_tokenizer,
    read_resources,
)
from nlp_text_classification.experiments.batch_scheduler import exclusive_queue
from nlp_text_classification.experiments.suite import (
    checked_file,
    load_training_data,
    load_yaml,
    read_json,
)
from nlp_text_classification.training.batch_run import Attempt, Tee
from nlp_text_classification.training.progress import Progress, write_atomic_json
from nlp_text_classification.training.resources import CPU_THREADS, GIB, protect

DATA_AUDIT = Path("configs/batch_b/tokenization_lock.json")
FORMAL = OUTPUT / "formal"
SOURCES = [
    "experiments/batch_b_s1.py", "experiments/batch_b_resources.py",
    "data/transformer_inputs.py", "training/transformer_runner.py", "training/batch_run.py",
    "evaluation/batch_metrics.py",
]


def _record(path, data):
    if path.exists():
        if read_json(path) != data:
            raise ValueError(f"Existing evidence differs: {path}")
    else:
        write_atomic_json(path, data)


def _audit_data(suite, datasets, resources, progress):
    data = {key: load_training_data(config) for key, config in datasets.items()}
    entries = {}
    for key, resource in resources.items():
        progress.emit("TOKENIZE", f"{key} auditing train and validation")
        tokenizer = load_tokenizer(Path(resource["cache"]))
        target = data[suite["models"][key]["dataset"]]
        checked_file(Path(target.settings["frozen_root"]) / "dataset_manifest.json",
                     target.settings["dataset_manifest_sha256"])
        entries[key] = audit_tokenization(tokenizer, target, suite["tokenization"], resource)
    record = {"suite": suite["suite_id"], "models": entries,
              "resource_lock_sha256": sha256_file(RESOURCE_LOCK)}
    _record(DATA_AUDIT, record)
    return record


def _identity(suite, datasets):
    root = Path(__file__).resolve().parents[1]
    return {
        "suite_sha256": sha256_file(SUITE),
        "config_sha256": digest_json({"suite": suite, "datasets": datasets}),
        "resource_lock_sha256": sha256_file(RESOURCE_LOCK),
        "tokenization_sha256": sha256_file(DATA_AUDIT),
        "sources": {name: sha256_file(root / name) for name in SOURCES},
    }


def _verify_attempt(run, data, *, rows=None):
    from sklearn.metrics import f1_score

    from nlp_text_classification.training.transformer_runner import (
        threshold_record,
        verify_prediction_file,
    )

    meta = read_json(run / "run_manifest.json")
    config = load_yaml(checked_file(run / "resolved_config.yaml", meta["config_sha256"]))
    result = read_json(run / "result.json")
    quality = read_json(run / "quality.json")
    bits = ("backbone_loaded", "fp32_weights", "finite_loss_and_gradients",
            "reloaded_predictions", "replayed_predictions")
    if result["status"] != "PASS" or quality["status"] != "PASS" or not all(
        quality.get(key) for key in bits
    ) or not all(quality["parameter_updates"].values()):
        raise ValueError(f"Failed training quality: {run}")
    if quality["splits_used"] != ["train", "validation"] or quality["label_order"] != label_names(data):
        raise ValueError(f"Training splits or label order changed: {run}")
    if (run / "test").exists() or quality["logit_width"] != (2 if data.binary else 28):
        raise ValueError(f"Unexpected test access or logit width: {run}")
    if rows is None:
        rows = {"train": data.train, "validation": data.validation}
    threshold = read_json(run / "thresholds.json")
    cutoff = threshold["threshold"]
    tolerance = config["selection"]["score_tie_atol"]
    for key, samples in rows.items():
        target = data.encode(samples)
        path = run / f"predictions_{key}.jsonl"
        metrics, scores = verify_prediction_file(path, samples, target, data.binary, cutoff,
                                                 auc=key == "validation")
        if metrics != read_json(run / f"metrics_{key}.json"):
            raise ValueError(f"Independent metric replay differs: {run} {key}")
        if key == "validation":
            candidates = threshold_record(target, scores, data.binary, config["selection"])
            validation_score = f1_score(target, scores >= cutoff, average="macro", zero_division=0,
                                        labels=[0, 1] if data.binary else None)
            if candidates != threshold or abs(validation_score - result["macro_f1"]) > tolerance:
                raise ValueError(f"Independent threshold or F1 replay differs: {run}")
    with (run / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    best, best_epoch = -1.0, 0
    settings = config["settings"]
    steps = math.ceil(len(rows["train"]) / settings["effective_batch_size"])
    for epoch, row in enumerate(history, 1):
        score = float(row["val_macro_f1"])
        if score > best + tolerance:
            best, best_epoch = score, epoch
        if (int(row["epoch"]), int(row["best_epoch"]), int(row["optimizer_steps"])) != (
            epoch, best_epoch, steps
        ) or (epoch < len(history) and epoch - best_epoch >= settings["patience"]):
            raise ValueError(f"Invalid epoch, update count or early stop: {run}")
    early = len(history) - best_epoch >= settings["patience"]
    if (best, best_epoch) != (result["macro_f1"], result["best_epoch"]) or (
        result["stop_reason"] != ("early_stopping" if early else "epoch_cap")
    ) or len(history) > settings["epochs"] or (not early and len(history) != settings["epochs"]):
        raise ValueError(f"Best epoch or stop decision differs: {run}")
    if float(history[best_epoch - 1]["threshold"]) != cutoff:
        raise ValueError(f"Checkpoint threshold differs from its history row: {run}")
    files = [
        "resolved_config.yaml", "run_manifest.json", "history.csv", "quality.json", "result.json",
        "thresholds.json", "metrics_train.json", "metrics_validation.json",
        "predictions_train.jsonl", "predictions_validation.jsonl", "checkpoint/config.json",
        "checkpoint/resume.pt", result["checkpoint"],
    ]
    return {"run": str(run), "result": result | {"epochs_completed": len(history)},
            "files": {name: sha256_file(run / name) for name in files}}


class Matrix:
    def __init__(self, suite, datasets, resources, audit, progress, *, smoke):
        import torch

        if not torch.cuda.is_available() or "RTX 4070 Ti SUPER" not in torch.cuda.get_device_name():
            raise RuntimeError("Batch B requires the RTX 4070 Ti SUPER")
        torch.set_num_threads(CPU_THREADS)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.suite, self.resources, self.audit, self.progress = suite, resources, audit, progress
        self.data = {key: load_training_data(item) for key, item in datasets.items()}
        self.smoke = smoke
        self.identity = _identity(suite, datasets)
        version = digest_json(self.identity)[:12]
        preflight = OUTPUT / "preflight-runs" / version
        self.output = preflight if smoke else FORMAL
        self.gate_path = preflight / "gate.json"
        self.output.mkdir(parents=True, exist_ok=True)
        self.hardware = {"gpu": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda,
                         "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp32"}
        self.queue_path = self.output / "queue.json"
        self.queue = read_json(self.queue_path) if self.queue_path.exists() else {
            "identity": self.identity, "hardware": self.hardware, "attempts": {}, "accepted": {},
            "adjustments": {},
        }
        if (self.queue["identity"], self.queue["hardware"]) != (self.identity, self.hardware):
            raise ValueError("Training source, inputs or GPU changed since queue creation")

    def _resolve(self, key, learning_rate, micro, checkpointing):
        experiment = self.suite["models"][key]
        settings = {**self.suite["training"], "learning_rate": learning_rate,
                    "micro_batch_size": micro, "evaluation_batch_size": min(32, micro * 2),
                    "gradient_checkpointing": checkpointing,
                    "precision": self.hardware["precision"]}
        if self.smoke:
            settings["epochs"] = 1
        if settings["effective_batch_size"] % micro:
            raise ValueError("Micro-batch must divide the effective batch")
        settings["accumulation_steps"] = settings["effective_batch_size"] // micro
        return {"suite_revision": self.suite["revision"], "experiment": {
                    **experiment, "id": key,
                }, "resource": self.resources[key], "settings": settings,
                "selection": self.suite["selection"], "quality": self.suite["quality"],
                "tokenization": self.audit[key], "inputs": self.identity, "hardware": self.hardware}

    def _attempt(self, slot, resolved):
        import torch

        from nlp_text_classification.training.transformer_runner import (
            prepare_partitions,
            train_transformer,
        )

        experiment = resolved["experiment"]
        key = experiment["id"]
        data = self.data[experiment["dataset"]]
        paths = self.queue["attempts"].setdefault(slot, [])
        previous = Path(paths[-1]) if paths else None
        resume = None
        if previous:
            result_file = previous / "result.json"
            if not result_file.exists():
                write_atomic_json(result_file, {"status": "INCOMPLETE", "reason": "Process exited"})
            saved = read_json(result_file)
            if load_yaml(previous / "resolved_config.yaml") == resolved:
                if saved["status"] == "PASS":
                    return previous
                if saved["status"] == "INCOMPLETE":
                    for item in reversed(paths):
                        candidate = Path(item)
                        if (candidate / "checkpoint/resume.pt").is_file() and (
                            load_yaml(candidate / "resolved_config.yaml") == resolved
                        ):
                            resume = candidate
                            break
        self.progress.context = {"slot": slot, "accepted_slots": len(self.queue["accepted"]),
                                 "total_slots": 5 if self.smoke else 8}
        attempt = Attempt(self.output / "runs", data, resolved, slot, self.suite["seed"],
                          self.progress, previous=str(previous) if previous else None, provenance={
                              "model_id": experiment["model_id"],
                              "model_revision": resolved["resource"]["revision"],
                              "tokenizer_sha256": self.audit[key]["tokenizer_sha256"],
                              "dataset_manifest_sha256": self.audit[key]["dataset_manifest_sha256"],
                              "precision": resolved["settings"]["precision"],
                          })
        paths.append(str(attempt.path))
        write_atomic_json(self.queue_path, self.queue)
        with (attempt.path / "stdout.log").open("x", encoding="utf-8") as out, (
            attempt.path / "stderr.log"
        ).open("x", encoding="utf-8") as err, redirect_stdout(Tee(sys.stdout, out)), (
            redirect_stderr(Tee(sys.stderr, err))
        ):
            try:
                attempt.event("START", f"{slot}, micro-batch {resolved['settings']['micro_batch_size']}")
                tokenizer = load_tokenizer(Path(resolved["resource"]["cache"]))
                rows, prepared = prepare_partitions(tokenizer, data, self.audit[key], smoke=self.smoke)
                details = train_transformer(attempt, data, rows, prepared, resolved, "cuda", resume)
                attempt.finish("PASS", f"validation Macro-F1 {details['macro_f1']:.5f}", **details)
            except KeyboardInterrupt:
                attempt.finish("INCOMPLETE", "Interrupted; completed epoch state retained")
                raise
            except Exception as exc:  # noqa: BLE001 - persist each failed attempt
                traceback.print_exc()
                attempt.finish("FAIL", str(exc), error_type=type(exc).__name__)
        gc.collect()
        torch.cuda.empty_cache()
        return attempt.path

    def _accept(self, slot, resolved, *, allow_resource_retry):
        from nlp_text_classification.training.transformer_runner import prepare_partitions

        key = resolved["experiment"]["id"]
        if slot in self.queue["accepted"]:
            entry = self.queue["accepted"][slot]
            for name, digest in entry["files"].items():
                checked_file(Path(entry["run"]) / name, digest)
            return entry
        data = self.data[resolved["experiment"]["dataset"]]
        micro = resolved["settings"]["micro_batch_size"]
        adjustments = self.queue["adjustments"]
        if slot in adjustments:
            micro = adjustments[slot]
            resolved = self._resolve(key, resolved["settings"]["learning_rate"], micro, True)
        for trial in range(2):
            run = self._attempt(slot, resolved)
            status = read_json(run / "result.json")
            if status["status"] == "PASS":
                break
            if trial or slot in adjustments or not allow_resource_retry or micro <= 1 or (
                status.get("error_type") != "OutOfMemoryError"
            ):
                raise RuntimeError(f"Encoder slot {slot} blocked. See {run / 'stderr.log'}")
            micro //= 2
            adjustments[slot] = micro
            write_atomic_json(self.queue_path, self.queue)
            self.progress.emit("RETRY", f"{slot} CUDA OOM, retry with micro-batch {micro}")
            resolved = self._resolve(key, resolved["settings"]["learning_rate"], micro, True)
        rows = None
        if self.smoke:
            tokenizer = load_tokenizer(Path(resolved["resource"]["cache"]))
            rows, _ = prepare_partitions(tokenizer, data, self.audit[key], smoke=True)
        accepted = _verify_attempt(run, data, rows=rows)
        self.queue["accepted"][slot] = accepted
        write_atomic_json(self.queue_path, self.queue)
        self.progress.emit("ACCEPTED", f"{slot} passed independent score replay",
                           accepted_slots=len(self.queue["accepted"]),
                           validation_macro_f1=accepted["result"]["macro_f1"])
        return accepted

    def preflight(self):
        rows = {}
        for key, model in self.suite["models"].items():
            config = self._resolve(key, self.suite["training"]["baseline_lr"],
                                   model["micro_batch_size"], model["gradient_checkpointing"])
            entry = self._accept(f"smoke/{key}", config, allow_resource_retry=True)
            candidate = model.get("preflight_micro_batch_candidate")
            if candidate and entry["result"]["peak_reserved_bytes"] <= 8 * GIB:
                trial_config = self._resolve(key, self.suite["training"]["baseline_lr"],
                                             candidate, False)
                trial_slot = f"smoke/{key}-throughput"
                try:
                    trial = self._accept(trial_slot, trial_config, allow_resource_retry=False)
                except RuntimeError:
                    last = Path(self.queue["attempts"][trial_slot][-1])
                    if read_json(last / "result.json").get("error_type") != "OutOfMemoryError":
                        raise
                    self.progress.emit("PREFLIGHT", f"{key} keeps the verified small micro-batch")
                else:
                    if trial["result"]["peak_reserved_bytes"] <= 10 * GIB:
                        entry = trial
            settings = load_yaml(Path(entry["run"]) / "resolved_config.yaml")["settings"]
            rows[key] = {"micro_batch_size": settings["micro_batch_size"],
                         "gradient_checkpointing": settings["gradient_checkpointing"], **entry}
        gate = {"identity": self.identity, "hardware": self.hardware, "models": rows}
        _record(self.gate_path, gate)
        self.progress.context = {}
        self.progress.emit("PREFLIGHT_COMPLETE", "4/4 encoder models accepted", gate=self.gate_path)
        return gate

    def train(self, gate):
        if gate["identity"] != self.identity or gate["hardware"] != self.hardware or (
            set(gate["models"]) != set(self.suite["models"])
        ):
            raise ValueError("The four-model CUDA preflight is stale")
        for entry in gate["models"].values():
            for name, digest in entry["files"].items():
                checked_file(Path(entry["run"]) / name, digest)
        for case in ("baseline", "alternate"):
            for key, model in self.suite["models"].items():
                preflight = gate["models"][key]
                learning_rate = (self.suite["training"]["baseline_lr"] if case == "baseline"
                                 else model["alternate_lr"])
                baseline_retry = self.queue["adjustments"].get(f"baseline/{key}")
                micro = baseline_retry or preflight["micro_batch_size"]
                settings = self._resolve(key, learning_rate, micro,
                                         preflight["gradient_checkpointing"] or bool(baseline_retry))
                self._accept(f"{case}/{key}", settings, allow_resource_retry=True)
        return self.queue


def execute(mode):
    from threadpoolctl import threadpool_limits

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with exclusive_queue(OUTPUT.parent / ".batch-a-gpu.lock"):
        progress = Progress(OUTPUT / "progress.md", shlex.join(sys.orig_argv),
                            title="Batch B S1 training progress")
        with threadpool_limits(limits=CPU_THREADS):
            suite, datasets, models = load_config()
            suite["models"] = models
            resources = read_resources(models, verify_weights=mode != "report")
            if mode == "data-audit":
                return _audit_data(suite, datasets, resources, progress)
            audit = read_json(DATA_AUDIT)
            if audit["resource_lock_sha256"] != sha256_file(RESOURCE_LOCK):
                raise ValueError("Tokenization audit belongs to another resource lock")
            from nlp_text_classification.evaluation.batch_b_report import publish

            if mode == "report":
                return publish(suite, datasets, audit["models"], read_json(FORMAL / "queue.json"),
                               progress)
            with protect(progress, process_limit=14 * GIB):
                matrix = Matrix(suite, datasets, resources, audit["models"], progress,
                                smoke=mode == "smoke")
                if mode == "smoke":
                    return matrix.preflight()
                queue = matrix.train(read_json(matrix.gate_path))
                report = publish(suite, datasets, audit["models"], queue, progress)
                progress.context = {}
                progress.emit("COMPLETE", "8/8 formal slots audited", **report)
                return report
