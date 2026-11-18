"""Run the validation-only GoEmotions training-budget supplement."""

from __future__ import annotations

import csv
import gc
import shlex
import shutil
import sys
from pathlib import Path

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.data.features import prepare_word2vec
from nlp_text_classification.experiments.batch_scheduler import exclusive_queue
from nlp_text_classification.experiments.final_evaluation import _check_ablation, _validate_run
from nlp_text_classification.experiments.suite import (
    checked_file,
    load_suite,
    load_training_data,
    load_yaml,
    read_json,
)
from nlp_text_classification.training.batch_run import run_attempt
from nlp_text_classification.training.progress import Progress, write_atomic_json
from nlp_text_classification.training.resources import CPU_THREADS, protect

PROTOCOL = Path("configs/experiment_suites/batch_a_s1.yaml")
OUTPUT = Path("artifacts/batch-a-s1")
COHORT = "batch-a-s1"
ABLATION = ("E2", "E5-TUNE", "E5-FROZEN")
CURVE_FIELDS = ("epoch", "loss", "val_macro_f1", "threshold", "grad_norm", "best_epoch")
RECOVERY_FILES = ("history.csv", "checkpoint/resume.pt")


def read_history(run):
    with (Path(run) / "history.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def same_state(left, right, torch):
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same_state(value, right[key], torch) for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            same_state(a, b, torch) for a, b in zip(left, right, strict=True)
        )
    return left == right


def compare_prefix(reference, replay, epochs):
    import torch

    original = read_history(reference)
    repeated = read_history(replay)
    trajectory = len(original) == len(repeated) == epochs and all(
        all(a[field] == b[field] for field in CURVE_FIELDS)
        for a, b in zip(original, repeated, strict=True)
    )
    left, right = (
        torch.load(run / "checkpoint/resume.pt", map_location="cpu", weights_only=True)
        for run in (reference, replay)
    )
    keys = ("epoch", "model", "optimizer", "sampler_rng", "cpu_rng", "cuda_rng",
            "best_epoch", "best_score", "best_checkpoint")
    state = {key: same_state(left[key], right[key], torch) for key in keys}
    old, new = (read_json(run / "result.json") for run in (reference, replay))
    identity = {key: old[key] == new[key] for key in (
        "vocabulary_sha256", "non_embedding_init_sha256", "initial_embedding_sha256",
        "word2vec", "torch", "cuda", "device",
    )}
    return {"epochs": epochs, "trajectory_exact": trajectory, "state": state, "identity": identity,
            "exact": trajectory and all(state.values()) and all(identity.values())}


class Supplement:
    def __init__(self, progress):
        self.progress = progress
        self.spec = load_yaml(PROTOCOL)
        self.dataset = self.spec["dataset"]
        self.seed = self.spec["primary_seed"]
        base, datasets = load_suite(Path(self.spec["base_suite"]))
        self.freeze_path = checked_file(Path(self.spec["base_freeze"]),
                                        self.spec["base_freeze_sha256"])
        self.frozen = load_yaml(self.freeze_path)
        settings = datasets[self.dataset]
        digest = self.frozen["dataset_manifest_sha256"][self.dataset]
        checked_file(Path(settings["frozen_root"]) / "dataset_manifest.json", digest)
        if (settings["dataset_manifest_sha256"], base["revision"]) != (
            digest, self.frozen["suite_revision"]
        ):
            raise ValueError("S1 data or base suite changed")
        self.data = load_training_data(settings)
        self.config = {**base, "execution": {**base["execution"],
                                           "artifacts_root": str(OUTPUT / "runs")}}
        self.originals = {}
        self.resolved = {}
        self.reference_files = {}
        for experiment in self.spec["experiments"]:
            entry = self.frozen["main"][f"{self.dataset}/{experiment}"]
            path = self._verify_entry(entry)
            resolved = load_yaml(path / "resolved_config.yaml")
            if resolved["settings"]["epoch_cap"] != self.spec["reference_epoch_cap"]:
                raise ValueError(f"Unexpected original epoch budget: {path}")
            self.resolved[experiment] = {
                **resolved, "settings": {**resolved["settings"],
                                         "epoch_cap": self.spec["epoch_cap"],
                                         "patience": self.spec["patience"]},
            }
            self.originals[experiment] = path
            self.reference_files[experiment] = {
                "run": str(path), "files": {name: sha256_file(path / name) for name in RECOVERY_FILES}
            }
        _check_ablation([self.originals[experiment] for experiment in ABLATION])
        anchor = self.frozen["main"][f"{self.dataset}/E1"]
        self._verify_entry(anchor)
        self.anchor = anchor["validation_macro_f1"]
        self.vector_sha = read_json(self.originals["E5-TUNE"] / "result.json")["word2vec"]["sha256"]
        self.queue_path = OUTPUT / "queue.json"
        self.inputs = {
            "protocol_sha256": sha256_file(PROTOCOL),
            "freeze_sha256": self.spec["base_freeze_sha256"],
            "training_manifest_sha256": settings["training_manifest_sha256"],
            "dataset_config_sha256": sha256_file(Path(base["dataset_configs"][self.dataset])),
            "word2vec_sha256": self.vector_sha,
            "base_runs": self.reference_files,
        }
        self.queue = (read_json(self.queue_path) if self.queue_path.exists() else
                      {"inputs": self.inputs, "attempts": {}, "audited": {}})
        if self.queue["inputs"] != self.inputs:
            raise ValueError("S1 protocol or original evidence changed since initialization")
        self.vectors = None

    @staticmethod
    def _verify_entry(entry):
        path = Path(entry["run"])
        for name, digest in entry["files"].items():
            checked_file(path / name, digest)
        return path

    def _word2vec(self):
        if self.vectors is None:
            identity = self.data.settings["training_manifest_sha256"]
            dimension = self.resolved["E5-TUNE"]["settings"]["embedding_dim"]
            name = f"{self.dataset}-{identity[:16]}-{dimension}d-{self.seed}.kv"
            source = checked_file(self.freeze_path.parent / "embeddings" / name, self.vector_sha)
            output = OUTPUT / "embeddings"
            output.mkdir(exist_ok=True)
            target = output / name
            if not target.exists():
                temp = target.with_suffix(".pending")
                shutil.copyfile(source, temp)
                checked_file(temp, self.vector_sha).replace(target)
            checked_file(target, self.vector_sha)
            self.vectors = prepare_word2vec(self.data, output, self.seed, dimension)
        return self.vectors

    def _attempt(self, experiment, seed, phase, parameters, start=None):
        import torch

        key = f"{phase}/{experiment}/{seed}"
        cohort = f"{COHORT}-{phase}"
        self.progress.context = {"task": key, "supplement": COHORT}
        audited = self.queue["audited"].get(key)
        if audited:
            return self._verify_entry(audited)
        attempts = self.queue["attempts"].setdefault(key, [])
        previous = Path(attempts[-1]) if attempts else None
        checkpoint = start
        run = None
        if previous:
            result_path = previous / "result.json"
            if not result_path.exists():
                write_atomic_json(result_path, {"status": "INCOMPLETE", "reason": "Process stopped"})
            status = read_json(result_path)["status"]
            if status == "PASS":
                run = previous
            elif status == "INCOMPLETE" and (previous / "checkpoint/resume.pt").is_file():
                checkpoint = previous

        def record(path):
            attempts.append(str(path))
            write_atomic_json(self.queue_path, self.queue)

        if run is None:
            run = run_attempt(
                self.data, parameters, OUTPUT / "runs", self.progress, cohort=cohort,
                seed=seed, device="cuda", word2vec=self._word2vec() if experiment in ABLATION[1:]
                else None, previous=str(previous or start) if previous or start else None,
                resume=checkpoint / "checkpoint" if checkpoint else None, on_start=record,
            )
            gc.collect()
            torch.cuda.empty_cache()
        entry = _validate_run(run, self.data, experiment, {cohort}, seed, self.config)
        if load_yaml(run / "resolved_config.yaml") != parameters:
            raise ValueError(f"Unexpected S1 run settings: {run}")
        entry["files"].update({name: sha256_file(run / name) for name in RECOVERY_FILES})
        self.queue["audited"][key] = entry
        write_atomic_json(self.queue_path, self.queue)
        return run

    def primary(self):
        comparison = {}
        runs = {}
        for experiment, resolved in self.resolved.items():
            epoch_cap = self.spec["reference_epoch_cap"]
            reference = {**resolved, "settings": {**resolved["settings"], "epoch_cap": epoch_cap}}
            replay = self._attempt(experiment, self.seed, "replay", reference)
            audit = compare_prefix(self.originals[experiment], replay, epoch_cap)
            comparison[experiment] = {"run": str(replay), **audit}
            write_atomic_json(OUTPUT / "prefix_audit.json", comparison)
            if not audit["exact"]:
                raise ValueError(f"Original 12-epoch prefix differs for {experiment}")
            run = self._attempt(experiment, self.seed, "primary", resolved, start=replay)
            if any(
                a != b for a, b in zip(read_history(run)[:epoch_cap], read_history(replay), strict=True)
            ):
                raise ValueError(f"Resumed history changed for {experiment}")
            runs[experiment] = str(run)
        _check_ablation([Path(runs[experiment]) for experiment in ABLATION])
        return runs

    def replicate(self, primary):
        runs = {}
        for seed in self.spec["replication_seeds"]:
            for experiment in primary["replicate"]:
                run = self._attempt(experiment, seed, "replication", self.resolved[experiment])
                runs[f"{experiment}/{seed}"] = str(run)
            if set(ABLATION) <= set(primary["replicate"]):
                _check_ablation([Path(runs[f"{experiment}/{seed}"]) for experiment in ABLATION])
        return runs

    def read_primary(self):
        primary = read_json(OUTPUT / "primary.json")
        if primary["inputs"] != self.inputs:
            raise ValueError("S1 primary decision belongs to another protocol")
        for experiment, run in primary["runs"].items():
            key = f"primary/{experiment}/{self.seed}"
            if self._verify_entry(self.queue["audited"][key]) != Path(run):
                raise ValueError(f"Primary run does not match S1 audit: {run}")
        return primary


def execute(mode):
    from threadpoolctl import threadpool_limits

    from nlp_text_classification.evaluation.batch_a_s1_report import publish

    with exclusive_queue(OUTPUT.parent / ".batch-a-gpu.lock"):
        progress = Progress(OUTPUT / "progress.md", shlex.join(sys.orig_argv))
        with threadpool_limits(limits=CPU_THREADS):
            if mode == "report":
                return publish(Supplement(progress), mode)
            with protect(progress):
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("S1 requires the Batch A CUDA device")
                progress.emit("S1", f"phase={mode}", gpu=torch.cuda.get_device_name())
                return publish(Supplement(progress), mode)
