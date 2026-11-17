"""Serial Batch A orchestration for baselines, Optuna search and replication."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution

from nlp_text_classification.data.features import prepare_word2vec
from nlp_text_classification.experiments.performance import review
from nlp_text_classification.experiments.suite import load_training_data, load_yaml, read_json
from nlp_text_classification.training.batch_run import resolve, run_attempt
from nlp_text_classification.training.progress import write_atomic_json


@contextmanager
def exclusive_queue(path):
    with path.open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"1")
            stream.flush()
        stream.seek(0)
        try:
            import msvcrt
        except ImportError:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        yield


def trial_space(config, experiment):
    hpo = config["hpo"]
    if experiment == "E1":
        search = hpo["svm_space"]
        groups = [f"{low}:{high}" for low, high in search["ngram_range"]]
        return {
            "c": FloatDistribution(*search["c_log_range"], log=True),
            "min_df": CategoricalDistribution(search["min_df"]),
            "ngram_range": CategoricalDistribution(groups),
            "class_weight": CategoricalDistribution(search["class_weight"]),
        }
    search = hpo["neural_space"]
    return {
        "learning_rate": FloatDistribution(*search["learning_rate_log_ranges"][experiment],
                                           log=True),
        "weight_decay": CategoricalDistribution(search["weight_decay"]),
        "dropout": CategoricalDistribution(search["dropout"]),
    }


def run_parameters(parameters):
    result = dict(parameters)
    grams = result.get("ngram_range")
    if isinstance(grams, str):
        result["ngram_range"] = [int(v) for v in grams.split(":")]
    return result


class Batch:
    def __init__(self, config, datasets, output, progress, device):
        self.config, self.datasets, self.output = config, datasets, output
        self.progress, self.device = progress, device
        self.runs = Path(config["execution"]["artifacts_root"])
        self.state_path = output / "queue.json"
        signature = hashlib.sha256(json.dumps({
            "config": config, "datasets": datasets, "device": device
        }, sort_keys=True).encode()).hexdigest()
        self.state = (read_json(self.state_path) if self.state_path.exists() else {
            "signature": signature, "tasks": {}, "proposals": {}, "retries": {"cpu": 0, "cuda": 0},
            "main": {}, "replications": {},
        })
        if self.state["signature"] != signature:
            raise ValueError("Queue configuration changed. Choose a new --output.")
        self.data = {key: load_training_data(dataset) for key, dataset in datasets.items()}
        self.vectors = {}
        self.save()

    def save(self):
        write_atomic_json(self.state_path, self.state)

    def attempt(self, dataset, experiment, cohort, seed, params=None,
                pruning=None, *, suffix=""):
        key = f"{cohort}/{dataset}/{experiment}/{seed}{suffix}"
        history = self.state["tasks"].setdefault(key, [])
        resumed = None
        device = "cpu" if experiment == "E1" else self.device
        if history:
            previous = Path(history[-1])
            result_path = previous / "result.json"
            status = read_json(result_path) if result_path.exists() else {"status": "INCOMPLETE"}
            if status["status"] in {"PASS", "PRUNED"}:
                return previous, status
            if len(history) > self.config["execution"]["retry_limit_per_attempt"]:
                return previous, status
            budget_key = "retry_budget_cpu" if device == "cpu" else "retry_budget_gpu"
            if self.state["retries"][device] >= self.config["execution"][budget_key]:
                return previous, status
            if status["status"] == "INCOMPLETE":
                checkpoint = previous / "checkpoint"
                if (checkpoint / "resume.pt").is_file() and (checkpoint / "model.pt").is_file():
                    resumed = checkpoint
            elif status.get("error_type") not in {"OSError", "PermissionError", "OutOfMemoryError"}:
                return previous, status
            if not result_path.exists():
                write_atomic_json(result_path, {"status": "INCOMPLETE", "reason": "Process stopped"})
        else:
            previous = None
        if experiment.startswith("E5") and dataset not in self.vectors:
            self.progress.emit("PREPARE", f"loading {dataset} Word2Vec")
            self.vectors[dataset] = prepare_word2vec(
                self.data[dataset], self.output / "embeddings", self.config["seed"],
                self.config["neural"]["embedding_dim"],
            )
        self.progress.context = {"task": key, "attempted_tasks": len(self.state["tasks"]),
                                 "main_selected": f"{len(self.state['main'])}/14"}
        kwargs = resolve(self.config, experiment, params)
        if resumed is not None:
            saved_config = load_yaml(previous / "resolved_config.yaml")
            if saved_config["suite_revision"] != kwargs["suite_revision"] or (
                saved_config["experiment"] != kwargs["experiment"]
            ):
                raise ValueError(f"Resume checkpoint uses another configuration: {previous}")
            kwargs = saved_config

        def record(path):
            if previous:
                self.state["retries"][device] += 1
            history.append(str(path))
            self.save()

        result = run_attempt(self.data[dataset], kwargs, self.runs, self.progress, cohort=cohort,
                             seed=seed, device=device, word2vec=self.vectors.get(dataset),
                             pruning=pruning, previous=str(previous) if previous else None,
                             resume=resumed, on_start=record)
        status = read_json(result / "result.json")
        if status.get("error_type") == "OutOfMemoryError":
            self.progress.emit("RESOURCE", "CUDA OOM. Retry budget still applies.",
                               failed_run=str(result))
            if device == "cuda":
                import torch

                torch.cuda.empty_cache()
        if status.get("error_type") in {"OSError", "PermissionError", "OutOfMemoryError"} and (
            len(history) == 1
        ):
            kwargs = dict(params or {})
            if status["error_type"] == "OutOfMemoryError" and experiment.startswith("E5"):
                return result, status
            if status["error_type"] == "OutOfMemoryError":
                base = resolve(self.config, experiment, kwargs)["settings"]
                new_micro = base.get("micro_batch_size", base["effective_batch_size"]) // 2
                if new_micro <= 0 or base["effective_batch_size"] % new_micro:
                    return result, status
                kwargs["micro_batch_size"] = new_micro
            return self.attempt(dataset, experiment, cohort, seed, kwargs, pruning, suffix=suffix)
        return result, status

    def baselines(self):
        result = {}
        seed = self.config["seed"]
        total = len(self.data) * len(self.config["experiments"])
        for dataset in self.data:
            for entry in self.config["experiments"]:
                experiment = entry["id"]
                self.progress.emit("BASELINE", f"{len(result) + 1}/{total} {dataset}/{experiment}",
                                   matrix_position=f"{len(result) + 1}/{total}")
                path, status = self.attempt(dataset, experiment, "baseline", seed)
                if status["status"] != "PASS":
                    raise RuntimeError(f"Baseline failure: {path}")
                result[(dataset, experiment)] = (path, status)
        return result

    def search(self, dataset, experiment, baseline):
        conf = self.config["hpo"]
        neural = experiment != "E1"
        startup = conf["neural_startup_trials" if neural else "svm_startup_trials"]
        median = conf["neural_pruner"]
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=median["n_startup_trials"],
            n_warmup_steps=median["first_check_epoch"], interval_steps=median["interval_epochs"],
        ) if neural else optuna.pruners.NopPruner()
        database = (self.output / "optuna.sqlite3").resolve().as_posix()
        study = optuna.create_study(
            study_name=f"{dataset}-{experiment}", storage=f"sqlite:///{database}",
            direction="maximize", load_if_exists=True, pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=conf["sampler_seed"], n_startup_trials=startup),
        )
        slots = conf["per_dataset_neural_model_trials" if neural else "per_dataset_svm_trials"]
        cutoff = conf["neural_min_accepted_trials"] if neural else 1
        extra = conf["neural_extension_trials"] if neural else 0
        accepted = 0
        candidates = [(-1, *baseline)]
        study_proposals = self.state["proposals"].setdefault(study.study_name, {})
        for slot in range(slots + extra):
            if slot >= slots and accepted >= cutoff:
                break
            key = str(slot)
            proposal = study_proposals.get(key)
            callback = None
            if proposal is None:
                for abandoned in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
                    study.tell(abandoned.number, state=optuna.trial.TrialState.FAIL)
                trial = study.ask(fixed_distributions=trial_space(self.config, experiment))
                proposal = {"number": trial.number, "params": run_parameters(trial.params),
                            "extension": slot >= slots}
                trial.set_user_attr("slot", slot)

                def pruning(epoch, value, trial=trial, allow=slot < slots):
                    trial.report(value, epoch)
                    return allow and trial.should_prune()

                if neural:
                    callback = pruning
                study_proposals[key] = proposal
                self.save()
            trial_number = proposal["number"]
            self.progress.emit("HPO", f"{study.study_name} candidate {slot + 1}/{slots + extra}",
                               accepted=accepted, initial_budget=slots, extension=slot >= slots)
            path, status = self.attempt(dataset, experiment, "hpo", self.config["seed"],
                                        proposal["params"], callback, suffix=f"/{slot}")
            proposal.update({"run": str(path), **status})
            self.save()
            if status["status"] == "PASS":
                accepted += 1
                study.tell(trial_number, status["macro_f1"], skip_if_finished=True)
                candidates.append((trial_number, path, status))
            else:
                optuna_status = (optuna.trial.TrialState.PRUNED if status["status"] == "PRUNED"
                                 else optuna.trial.TrialState.FAIL)
                study.tell(trial_number, state=optuna_status, skip_if_finished=True)
        if accepted < cutoff:
            raise RuntimeError(f"{study.study_name}: {accepted} accepted trials, required {cutoff}")
        best = max(item[2]["macro_f1"] for item in candidates)
        tolerance = self.config["selection"]["score_tie_atol"]
        eligible = [item for item in candidates if best - item[2]["macro_f1"] <= tolerance]
        _, path, _ = min(eligible, key=lambda item: item[0])
        return path

    def select_main(self, baselines):
        for dataset in self.data:
            for item in self.config["experiments"]:
                experiment = item["id"]
                key = f"{dataset}/{experiment}"
                if key in self.state["main"]:
                    continue
                if experiment.startswith("E5"):
                    cnn = Path(self.state["main"][f"{dataset}/E2"])
                    params = load_yaml(cnn / "resolved_config.yaml")["settings"]
                    if params == self.config["neural"]:
                        path = baselines[(dataset, experiment)][0]
                    else:
                        path, status = self.attempt(dataset, experiment, "ablation",
                                                    self.config["seed"], params)
                        if status["status"] != "PASS":
                            raise RuntimeError(f"Ablation failure: {path}")
                else:
                    path = self.search(dataset, experiment, baselines[(dataset, experiment)])
                self.state["main"][key] = str(path)
                self.save()
                self.progress.emit("SELECT", f"{key} selected {path.name}",
                                   main_selected=f"{len(self.state['main'])}/14", run=path)

    def best_recurrent(self, dataset):
        keys = self.config["replication"]["select_best_of_per_dataset"]
        scores = {key: read_json(Path(self.state["main"][f"{dataset}/{key}"]) /
                                 "result.json")["macro_f1"] for key in keys}
        best = max(scores.values())
        tol = self.config["selection"]["score_tie_atol"]
        return next(key for key in keys if best - scores[key] <= tol), scores

    def replicate(self, dataset, experiment, seeds, *, optional=False):
        main = Path(self.state["main"][f"{dataset}/{experiment}"])
        params = load_yaml(main / "resolved_config.yaml")["settings"]
        store = self.state.setdefault("optional_replications", {}) if optional else (
            self.state["replications"]
        )
        scores = []
        for seed in seeds:
            role = "extra_replication" if optional else "replication"
            self.progress.emit("REPLICATION", f"{dataset}/{experiment} seed={seed}",
                               required_completed=f"{len(self.state['replications'])}/30")
            path, status = self.attempt(dataset, experiment, role, seed, params)
            store[f"{dataset}/{experiment}/{seed}"] = str(path)
            self.save()
            if status["status"] != "PASS":
                if not optional:
                    raise RuntimeError(f"Replication failure: {path}")
            else:
                scores.append(status["macro_f1"])
        return scores

    def replicate_all(self):
        spec = self.config["replication"]
        for dataset in self.data:
            best, scores = self.best_recurrent(dataset)
            table = {}
            for experiment in [*spec["required_experiments_per_dataset"], best]:
                sample = self.replicate(dataset, experiment, spec["seeds"])
                table[experiment] = float(np.std(sample, ddof=1))
            least_stable = max(table, key=table.get)
            if table[least_stable] >= spec["high_sd_macro_f1"]:
                self.replicate(dataset, least_stable, spec["extra_seeds_for_high_sd"], optional=True)
            else:
                remaining = {key: value for key, value in scores.items() if key != best}
                close = max(remaining, key=remaining.get)
                if scores[best] - remaining[close] <= spec["near_tie_macro_f1"]:
                    self.replicate(dataset, close, spec["seeds"], optional=True)


def run_batch(config, datasets, output, progress, device):
    if device != "cuda":
        raise ValueError("Formal Batch A requires the verified CUDA device")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gate_path = output.parent / "batch-a-smoke" / "summary.json"
    gate = read_json(gate_path)
    if (gate["passed"], gate["device"], gate["suite_revision"]) != (14, "cuda", config["revision"]):
        raise ValueError("The formal batch requires the complete CUDA 14/14 smoke gate")
    with exclusive_queue(output.parent / ".batch-a-gpu.lock"):
        batch = Batch(config, datasets, output, progress, device)
        batch.select_main(batch.baselines())
        batch.replicate_all()
        review(batch)
        progress.context = {}
        progress.emit("READY_FOR_AUDIT", "Validation training and replication complete",
                      main=len(batch.state["main"]), replications=len(batch.state["replications"]))
        return {"queue": str(batch.state_path), "main": len(batch.state["main"]),
                "replications": len(batch.state["replications"])}
