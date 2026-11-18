"""Run Batch B's fixed seed replications and DistilBERT budget comparison."""

from __future__ import annotations

import math
import shlex
import sys
from pathlib import Path

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.experiments.batch_b_resources import load_config, read_resources
from nlp_text_classification.experiments.batch_b_s1 import (
    DATA_AUDIT,
    FORMAL,
    Matrix,
    _identity,
    _record,
    _verify_attempt,
)
from nlp_text_classification.experiments.batch_scheduler import exclusive_queue
from nlp_text_classification.experiments.suite import (
    checked_file,
    load_training_data,
    load_yaml,
    read_json,
)
from nlp_text_classification.training.progress import Progress
from nlp_text_classification.training.resources import CPU_THREADS, GIB, protect

OUTPUT = Path("artifacts/batch-b-s2")
SPEC = Path("configs/experiment_suites/batch_b_s2.yaml")
PROBE = "budget/B1/42"
TOTAL = 9


def budget_decision(short_f1, long_f1, minimum, tolerance):
    gain = long_f1 - short_f1
    return {"s1_macro_f1": short_f1, "s2_macro_f1": long_f1, "gain": gain,
            "minimum_macro_f1_gain": minimum,
            "selected_budget": "s2_12_epochs" if gain + tolerance >= minimum else "s1_6_epochs"}


def check_files(entry):
    run = Path(entry["run"])
    for file, digest in entry["files"].items():
        checked_file(run / file, digest)
    return run


class Inputs:
    def __init__(self):
        self.spec = load_yaml(SPEC)
        self.seeds = self.spec["seeds"]
        self.suite, self.datasets, self.models = load_config()
        self.suite["models"] = self.models
        self.results_path = checked_file(Path(self.spec["s1_results"]), self.spec["s1_results_sha256"])
        self.delivery_path = checked_file(Path(self.spec["s1_delivery_audit"]),
                                          self.spec["s1_delivery_audit_sha256"])
        self.delivery = read_json(self.delivery_path)
        self.s1 = read_json(self.results_path)
        self.s1_identity = _identity(self.suite, self.datasets)
        if self.s1["identity"] != self.s1_identity:
            raise ValueError("Batch B S1 trainer or data protocol differs from the archived identity")
        if self.delivery["status"] != "PASS" or self.s1["selected"] != self.delivery["selected"]:
            raise ValueError("Unaccepted or inconsistent S1 model selection")
        self.formal_path = checked_file(FORMAL / "audit.json", self.delivery["formal_audit_sha256"])
        self.formal = read_json(self.formal_path)
        checked_file(FORMAL / "queue.json", self.formal["queue_sha256"])
        if (self.formal["status"], self.formal["selected"], self.formal["splits_used"]) != (
            "PASS", self.s1["selected"], ["train", "validation"]
        ) or len(self.s1["runs"]) != 8:
            raise ValueError("S1 audit does not describe all accepted validation slots")
        self.audit = read_json(DATA_AUDIT)["models"]
        self.data = {key: load_training_data(settings) for key, settings in self.datasets.items()}
        self.originals = {}
        self.configs = {}
        self.identity = {
            "suite": self.spec["suite_id"], "revision": self.spec["revision"],
            "protocol_sha256": sha256_file(SPEC), "s1_training_identity": self.s1_identity,
            "s1_results_sha256": sha256_file(self.results_path),
            "s1_delivery_audit_sha256": sha256_file(self.delivery_path),
            "s2_engine_sha256": sha256_file(Path(__file__)),
        }
        for key, slot in self.s1["selected"].items():
            source = self.s1["runs"][slot]
            run = check_files(source)
            config = load_yaml(run / "resolved_config.yaml")
            manifest = read_json(run / "run_manifest.json")
            if source["files"] != self.formal["runs"][slot] or (
                manifest["seed"], manifest["experiment"], config["inputs"], config["settings"]
            ) != (self.seeds[0], key, self.s1_identity, source["settings"]):
                raise ValueError(f"S1 selected run differs from audited configuration: {key}")
            if config["tokenization"] != self.audit[key] or config["selection"] != self.suite["selection"]:
                raise ValueError(f"S1 tokenizer or metric protocol differs: {key}")
            self.originals[key] = source
            self.configs[key] = config
        if set(self.originals) != set(self.models) or self.seeds != [42, 123, 2026]:
            raise ValueError("S2 requires all four S1 models and the approved seed list")

    def audit_preservation(self, *, all_attempts):
        for file, digest in self.delivery["reports"].items():
            checked_file(Path(file), digest)
        checked_file(self.formal_path, self.delivery["formal_audit_sha256"])
        checked_file(FORMAL / "queue.json", self.formal["queue_sha256"])
        if all_attempts:
            for slot, entry in self.s1["runs"].items():
                if entry["files"] != self.formal["runs"][slot]:
                    raise ValueError(f"S1 slot fingerprints disagree: {slot}")
                check_files(entry)

    def resolved(self, key, seed, *, extended=False):
        config = self.configs[key]
        settings = dict(config["settings"])
        if extended:
            budget = self.spec["b1_budget"]
            settings.update(epochs=budget["epochs"], patience=budget["patience"])
        return {**config, "suite_revision": self.spec["revision"], "inputs": self.identity,
                "settings": settings, "s2": {"seed": seed, "s1_run": self.originals[key]["run"],
                                              "budget": "s2_12_epochs" if extended else "s1_6_epochs"}}

    def choose_b1(self, probe):
        budget = self.spec["b1_budget"]
        decision = budget_decision(
            self.originals["B1"]["result"]["macro_f1"], probe["result"]["macro_f1"],
            budget["minimum_macro_f1_gain"], self.suite["selection"]["score_tie_atol"],
        )
        options = {"s1_6_epochs": self.originals["B1"], "s2_12_epochs": probe}
        decision["runs"] = {key: entry["run"] for key, entry in options.items()}
        decision["selected_run"] = options[decision["selected_budget"]]["run"]
        return decision


class S2Progress(Progress):
    def _emit(self, phase, message, **details):
        super()._emit(phase, message, **(details | {"total_slots": TOTAL}))


class Replications(Matrix):
    def __init__(self, inputs, progress):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Batch B S2 requires the S1 CUDA device")
        self.hardware = {"gpu": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda,
                         "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp32"}
        if self.hardware != inputs.s1["hardware"]:
            raise ValueError("S2 hardware or precision differs from S1")
        torch.set_num_threads(CPU_THREADS)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.inputs, self.output, self.progress = inputs, OUTPUT, progress
        self.data, self.audit, self.smoke = inputs.data, inputs.audit, False
        self.suite = dict(inputs.suite)
        self.queue_path = OUTPUT / "queue.json"
        self.queue = read_json(self.queue_path) if self.queue_path.exists() else {
            "identity": inputs.identity, "hardware": self.hardware, "attempts": {}, "accepted": {},
            "adjustments": {},
        }
        if (self.queue["identity"], self.queue["hardware"]) != (inputs.identity, self.hardware):
            raise ValueError("S2 queue identity or device changed")
        if self.queue["adjustments"]:
            raise ValueError("S2 replications require the exact selected S1 batch settings")

    def _run_slot(self, slot, key, seed, *, extended=False):
        resolved = self.inputs.resolved(key, seed, extended=extended)
        self.suite["seed"] = seed
        entry = self._accept(slot, resolved, allow_resource_retry=False)
        verify_config(entry, slot, seed, resolved, self.data[resolved["experiment"]["dataset"]])
        return entry

    def run(self):
        probe = self._run_slot(PROBE, "B1", self.inputs.seeds[0], extended=True)
        decision = self.inputs.choose_b1(probe)
        _record(OUTPUT / "b1-selection.json", decision)
        self.progress.context = {}
        self.progress.emit("B1_SELECTED", decision["selected_budget"], gain=decision["gain"],
                           accepted_slots=len(self.queue["accepted"]))
        for key in self.inputs.models:
            extended = key == "B1" and decision["selected_budget"] == "s2_12_epochs"
            for seed in self.inputs.seeds[1:]:
                self._run_slot(f"replication/{key}/{seed}", key, seed, extended=extended)
        return self.queue


def verify_config(entry, slot, seed, resolved, data):
    run = Path(entry["run"])
    if not run.resolve().is_relative_to((OUTPUT / "runs").resolve()):
        raise ValueError(f"Run outside the S2 namespace: {run}")
    manifest = read_json(run / "run_manifest.json")
    if (manifest["cohort"], manifest["seed"]) != (slot, seed) or (
        load_yaml(run / "resolved_config.yaml") != resolved
    ):
        raise ValueError(f"Unexpected S2 seed, cohort or resolved configuration: {run}")
    expected = resolved["settings"]
    quality = read_json(run / "quality.json")
    steps = math.ceil(len(data.train) / expected["effective_batch_size"]) * expected["epochs"]
    if (quality["scheduler"], quality["planned_steps"], quality["warmup_steps"]) != (
        expected["scheduler"], steps, math.ceil(steps * expected["warmup_ratio"])
    ):
        raise ValueError(f"Unexpected learning-rate schedule: {run}")


def completed_runs(inputs, queue, progress):
    expected = {PROBE, *(f"replication/{key}/{seed}" for key in inputs.models
                         for seed in inputs.seeds[1:])}
    if set(queue["accepted"]) != expected or queue["identity"] != inputs.identity:
        raise ValueError("S2 report requires the budget probe and eight accepted seed replications")
    probe = queue["accepted"][PROBE]
    decision = inputs.choose_b1(probe)
    if read_json(OUTPUT / "b1-selection.json") != decision:
        raise ValueError("S2 budget decision changed")
    for slot, entry in queue["accepted"].items():
        _, key, seed = slot.split("/")
        seed = int(seed)
        extended = slot == PROBE or (key == "B1" and decision["selected_budget"] == "s2_12_epochs")
        resolved = inputs.resolved(key, seed, extended=extended)
        data = inputs.data[resolved["experiment"]["dataset"]]
        progress.emit("AUDIT", f"replaying {slot}", accepted_slots=TOTAL)
        if _verify_attempt(Path(entry["run"]), data) != entry:
            raise ValueError(f"S2 evidence changed since acceptance: {slot}")
        verify_config(entry, slot, seed, resolved, data)
    return decision


def execute(mode):
    from threadpoolctl import threadpool_limits

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with exclusive_queue(OUTPUT.parent / ".batch-a-gpu.lock"):
        progress = S2Progress(OUTPUT / "progress.md", shlex.join(sys.orig_argv),
                              title="Batch B S2 training progress")
        with threadpool_limits(limits=CPU_THREADS):
            inputs = Inputs()
            inputs.audit_preservation(all_attempts=mode == "audit")
            if mode in {"audit", "run"}:
                read_resources(inputs.models)
            if mode == "audit":
                status = {"status": "PASS", "identity": inputs.identity,
                          "s1_slots_verified": len(inputs.s1["runs"]), "new_slots": TOTAL}
                _record(OUTPUT / "input-audit.json", status)
                progress.emit("INPUTS_AUDITED", "S1 evidence and offline resources verified")
                return status
            if mode == "run":
                gate = read_json(OUTPUT / "input-audit.json")
                if gate["status"] != "PASS" or gate["identity"] != inputs.identity:
                    raise ValueError("S2 input audit is stale")
                with protect(progress, process_limit=14 * GIB):
                    queue = Replications(inputs, progress).run()
            else:
                queue = read_json(OUTPUT / "queue.json")
            progress.context = {}
            decision = completed_runs(inputs, queue, progress)
            inputs.audit_preservation(all_attempts=True)
            from nlp_text_classification.evaluation.batch_b_s2_report import publish

            result = publish(inputs, queue, decision)
            progress.emit("COMPLETE", "9/9 new slots audited; validation-only freeze proposal ready",
                          accepted_slots=TOTAL, **result)
            return result
