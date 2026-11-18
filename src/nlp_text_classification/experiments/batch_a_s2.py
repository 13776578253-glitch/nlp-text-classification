"""Continue S1 trajectories that are still improving near the training cap."""

from __future__ import annotations

import gc
import shlex
import sys
from pathlib import Path
from statistics import linear_regression, mean

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.experiments.batch_a_s1 import (
    CURVE_FIELDS,
    RECOVERY_FILES,
    Supplement,
    read_history,
    same_state,
)
from nlp_text_classification.experiments.batch_a_s1 import (
    OUTPUT as S1_OUTPUT,
)
from nlp_text_classification.experiments.batch_scheduler import exclusive_queue
from nlp_text_classification.experiments.final_evaluation import _validate_run
from nlp_text_classification.experiments.suite import checked_file, load_yaml, read_json
from nlp_text_classification.training.batch_run import run_attempt
from nlp_text_classification.training.progress import Progress, write_atomic_json
from nlp_text_classification.training.resources import CPU_THREADS, protect

PROTOCOL = Path("configs/experiment_suites/batch_a_s2.yaml")
OUTPUT = Path("artifacts/batch-a-s2")
RESUME = "checkpoint/resume.pt"
STATE_FIELDS = {
    "epoch", "model", "optimizer", "sampler_rng", "cpu_rng", "cuda_rng",
    "best_epoch", "best_score", "best_checkpoint",
}


def trajectory(run, cap, width):
    result = read_json(Path(run) / "result.json")
    history = read_history(run)
    tail = [{field: float(row[field]) for field in ("val_macro_f1", "loss")} | {
        "epoch": int(row["epoch"]),
    } for row in history[-2 * width:]]
    slope, mean_gain = None, None
    if len(tail) == 2 * width:
        previous, recent = tail[:width], tail[width:]
        scores = [row["val_macro_f1"] for row in recent]
        slope = linear_regression([row["epoch"] for row in recent], scores).slope
        mean_gain = mean(scores) - mean(row["val_macro_f1"] for row in previous)
    near_cap = cap - width + 1 <= result["best_epoch"] <= cap
    rising = (len(history) == cap and near_cap and slope is not None and mean_gain is not None
              and slope > 0 and mean_gain > 0)
    return {
        "run": str(run), "macro_f1": result["macro_f1"], "best_epoch": result["best_epoch"],
        "epochs": len(history), "stop_reason": result["stop_reason"], "tail": tail,
        "tail_slope": slope, "tail_mean_gain": mean_gain, "rising_at_cap": rising,
    }


class BoundaryExperiment:
    def __init__(self, progress):
        self.progress = progress
        self.spec = load_yaml(PROTOCOL)
        self.s1 = Supplement(progress)
        self.seeds = [self.s1.seed, *self.s1.spec["replication_seeds"]]
        self.s1_cap = self.s1.spec["epoch_cap"]
        self.tolerance = self.s1.config["selection"]["score_tie_atol"]
        result_path = checked_file(Path(self.spec["s1_results"]), self.spec["s1_results_sha256"])
        audit_path = checked_file(Path(self.spec["s1_delivery_audit"]),
                                  self.spec["s1_delivery_audit_sha256"])
        audit = read_json(audit_path)
        s1_queue = S1_OUTPUT / "queue.json"
        checked_file(s1_queue, audit["files"][s1_queue.as_posix()])
        if audit["status"] != "PASS" or self.spec["patience"] != self.s1.spec["patience"]:
            raise ValueError("S2 requires audited S1 results and the same patience")
        results = read_json(result_path)
        self.entries = {}
        for model in self.s1.resolved:
            for seed in self.seeds:
                key = f"{model}/{seed}"
                primary = seed == self.s1.seed
                run = results["after"][model] if primary else results["replications"][key]["extended"]
                phase = "primary" if primary else "replication"
                entry = self.s1.queue["audited"][f"{phase}/{key}"]
                if entry["run"] != run["run"] or entry["validation_macro_f1"] != run["macro_f1"]:
                    raise ValueError(f"S1 results disagree with its audited record: {key}")
                self.entries[key] = entry
        self.inputs = {"protocol_sha256": sha256_file(PROTOCOL),
                       "s1_results_sha256": sha256_file(result_path),
                       "s1_delivery_audit_sha256": sha256_file(audit_path)}
        self.config = {**self.s1.config, "execution": {**self.s1.config["execution"],
                                                      "artifacts_root": str(OUTPUT / "runs")}}
        self.queue_path = OUTPUT / "queue.json"
        self.queue = read_json(self.queue_path) if self.queue_path.exists() else {
            "inputs": self.inputs, "attempts": {}, "audited": {},
        }
        if self.queue["inputs"] != self.inputs:
            raise ValueError("S2 protocol or S1 evidence changed")
        self.vectors = None

    def screen(self):
        records = {}
        candidates = {}
        width = self.spec["tail_window"]
        for key, entry in self.entries.items():
            run = Path(entry["run"])
            for name, digest in entry["files"].items():
                if name != RESUME or (run / name).exists():
                    checked_file(run / name, digest)
            item = trajectory(run, self.s1_cap, width)
            records[key] = item
            if item["rising_at_cap"]:
                model, seed = key.split("/")
                candidates.setdefault(model, []).append(int(seed))
        payload = {"inputs": self.inputs, "records": records, "candidates": candidates}
        path = OUTPUT / "screening.json"
        if path.exists() and read_json(path) != payload:
            raise ValueError("The published S2 candidate screen changed")
        write_atomic_json(path, payload)
        return payload

    def _word2vec(self):
        if self.vectors is None:
            from gensim.models import KeyedVectors

            old = read_json(self.s1.originals["E5-TUNE"] / "result.json")["word2vec"]
            identity = self.s1.data.settings["training_manifest_sha256"][:16]
            name = f'{self.s1.dataset}-{identity}-{old["dimension"]}d-{self.s1.seed}.kv'
            source = checked_file(S1_OUTPUT / "embeddings" / name, self.s1.vector_sha)
            vectors = KeyedVectors.load(str(source))
            meta = {field: old[field] for field in ("source", "sha256", "dimension", "vectors")}
            self.vectors = vectors, meta
        return self.vectors

    def _attempt(self, model, seed, phase, resolved, source=None):
        import torch

        key = f"{phase}/{model}/{seed}"
        self.progress.context = {"task": key, "supplement": "batch-a-s2"}
        audited = self.queue["audited"].get(key)
        if audited:
            return self.s1._verify_entry(audited)
        attempts = self.queue["attempts"].setdefault(key, [])
        previous = Path(attempts[-1]) if attempts else None
        checkpoint = source
        run = None
        if previous:
            result = previous / "result.json"
            if not result.exists():
                write_atomic_json(result, {"status": "INCOMPLETE", "reason": "Process stopped"})
            status = read_json(result)["status"]
            if status == "PASS":
                run = previous
            elif status == "INCOMPLETE" and (previous / RESUME).is_file():
                checkpoint = previous

        def record(path):
            attempts.append(str(path))
            write_atomic_json(self.queue_path, self.queue)

        cohort = f"batch-a-s2-{phase}"
        if run is None:
            run = run_attempt(
                self.s1.data, resolved, OUTPUT / "runs", self.progress,
                cohort=cohort, seed=seed, device="cuda", on_start=record,
                previous=str(previous or source) if previous or source else None,
                resume=checkpoint / "checkpoint" if checkpoint else None,
                word2vec=self._word2vec() if model.startswith("E5-") else None,
            )
            gc.collect()
            torch.cuda.empty_cache()
        entry = _validate_run(run, self.s1.data, model, {cohort}, seed, self.config)
        if load_yaml(run / "resolved_config.yaml") != resolved:
            raise ValueError(f"S2 run configuration changed: {run}")
        entry["files"].update({name: sha256_file(run / name) for name in RECOVERY_FILES})
        self.queue["audited"][key] = entry
        write_atomic_json(self.queue_path, self.queue)
        return run

    def _source(self, model, seed, old):
        import torch

        reference = Path(old["run"])
        checkpoint = reference / RESUME
        if checkpoint.is_file():
            entry = self.entries[f"{model}/{seed}"]
            checked_file(checkpoint, entry["files"][RESUME])
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if STATE_FIELDS <= state.keys() and all(state[key] is not None for key in STATE_FIELDS):
                optimizer = state["optimizer"]
                if optimizer.get("state") and optimizer.get("param_groups") and (
                    state["epoch"], state["best_epoch"], state["best_score"]
                ) == (old["epochs"], old["best_epoch"], old["macro_f1"]):
                    return reference, "s1_checkpoint"
        replay = self._attempt(model, seed, "replay", self.s1.resolved[model])
        before, after = read_history(reference), read_history(replay)
        if len(before) != len(after) or any(
            any(a[field] != b[field] for field in CURVE_FIELDS)
            for a, b in zip(before, after, strict=True)
        ):
            raise ValueError(f"S2 fallback did not reproduce the S1 trajectory: {model}/{seed}")
        left, right = (torch.load(path / "checkpoint/model.pt", map_location="cpu", weights_only=True)
                       for path in (reference, replay))
        if not same_state(left, right, torch):
            raise ValueError(f"S2 fallback did not reproduce the S1 selected checkpoint: {model}/{seed}")
        return replay, "verified_replay"

    def _extend(self, model, seed, old):
        source, recovery = self._source(model, seed, old)
        s1_resolved = self.s1.resolved[model]
        resolved = {**s1_resolved, "settings": {
            **s1_resolved["settings"], "epoch_cap": self.spec["epoch_cap"],
        }}
        run = self._attempt(model, seed, "extension", resolved, source=source)
        before = read_history(source)
        after = read_history(run)
        if after[:len(before)] != before:
            raise ValueError(f"The S2 continuation changed its validated prefix: {model}/{seed}")
        new = trajectory(run, self.spec["epoch_cap"], self.spec["tail_window"])
        return {"s1": old, "s2": new, "gain": new["macro_f1"] - old["macro_f1"],
                "source": str(source), "source_epoch": len(before),
                "source_checkpoint_sha256": sha256_file(source / RESUME),
                "prefix_exact": True, "recovery": recovery}

    def run(self, screen):
        source = screen["records"]
        results = {}
        for model in screen["candidates"]:
            key = f"{model}/{self.seeds[0]}"
            results[key] = self._extend(model, self.seeds[0], source[key])
        threshold = self.spec["minimum_primary_gain"]
        replicate = [model for model in screen["candidates"]
                     if results[f"{model}/{self.seeds[0]}"]["gain"] + self.tolerance >= threshold]
        decision = {"inputs": self.inputs, "primary": results.copy(), "replicate": replicate}
        decision_path = OUTPUT / "primary.json"
        if decision_path.exists() and read_json(decision_path) != decision:
            raise ValueError("The published S2 replication decision changed")
        write_atomic_json(decision_path, decision)
        self.progress.context = {}
        self.progress.emit("S2_REPLICATION_GATE", f"models={', '.join(replicate) or 'none'}",
                           gain_required=threshold)
        for model in replicate:
            for seed in self.seeds[1:]:
                key = f"{model}/{seed}"
                results[key] = self._extend(model, seed, source[key])
        data = {"screen": screen, "runs": results, "replicate": replicate,
                "recommend_more_budget": [model for model in screen["candidates"]
                                          if any(row["s2"]["rising_at_cap"] for key, row
                                                 in results.items() if key.startswith(model + "/"))]}
        write_atomic_json(OUTPUT / "results.json", data)
        return data

    def saved_results(self, screen):
        data = read_json(OUTPUT / "results.json")
        if data["screen"] != screen:
            raise ValueError("S2 results and candidate screen disagree")
        for key, item in data["runs"].items():
            entry = self.queue["audited"][f"extension/{key}"]
            self.s1._verify_entry(entry)
            checked_file(Path(item["source"]) / RESUME, item["source_checkpoint_sha256"])
        return data


def execute(mode):
    from threadpoolctl import threadpool_limits

    from nlp_text_classification.evaluation.batch_a_s2_report import publish

    with exclusive_queue(OUTPUT.parent / ".batch-a-gpu.lock"):
        progress = Progress(OUTPUT / "progress.md", shlex.join(sys.orig_argv))
        with threadpool_limits(limits=CPU_THREADS):
            if mode == "run":
                with protect(progress):
                    import torch

                    if not torch.cuda.is_available():
                        raise RuntimeError("S2 requires the Batch A CUDA device")
                    experiment = BoundaryExperiment(progress)
                    screening = experiment.screen()
                    data = experiment.run(screening)
            else:
                experiment = BoundaryExperiment(progress)
                screening = experiment.screen()
                data = experiment.saved_results(screening) if mode == "report" else None
            published = publish(experiment, screening, data)
            progress.context = {}
            progress.emit("S2_COMPLETE" if data else "S2_SCREENED",
                          f'{len(screening["candidates"])} candidate models', **published)
            return published
