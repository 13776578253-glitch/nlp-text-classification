"""Independently audit validation evidence and evaluate only frozen Batch A runs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import f1_score

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.evaluation import batch_metrics
from nlp_text_classification.experiments.batch_scheduler import exclusive_queue
from nlp_text_classification.experiments.batch_svm import predict_model as svm_predict
from nlp_text_classification.experiments.suite import (
    checked_file,
    load_partition,
    load_training_data,
    load_yaml,
    read_json,
)
from nlp_text_classification.training.progress import write_atomic_json


def _checkpoint(run, experiment):
    return run / "checkpoint" / ("model.joblib" if experiment == "E1" else "model.pt")


def _evaluator_sources():
    source = Path(__file__).resolve().parents[1]
    return {
        "metrics": Path(batch_metrics.__file__),
        "audit": Path(__file__),
        "neural_runner": source / "training" / "neural_runner.py",
        "neural_model": source / "models" / "neural.py",
        "svm": source / "experiments" / "batch_svm.py",
        "features": source / "data" / "features.py",
    }


def _validate_run(run, data, experiment, cohort, seed, config):
    metadata = read_json(run / "run_manifest.json")
    result = read_json(run / "result.json")
    quality = read_json(run / "quality.json")
    resolved_path = checked_file(run / "resolved_config.yaml", metadata["config_sha256"])
    resolved = load_yaml(resolved_path)
    threshold = read_json(run / "thresholds.json")
    validation = read_json(run / "metrics_validation.json")
    predictions = run / "predictions_validation.jsonl"
    ids, y, scores, y_pred = batch_metrics.read_predictions(predictions)
    if not run.resolve().is_relative_to(Path(config["execution"]["artifacts_root"]).resolve()):
        raise ValueError(f"Run outside trusted artifacts: {run}")
    if (metadata["dataset"], metadata["experiment"], metadata["seed"]) != (
        data.manifest["dataset"], experiment, seed
    ):
        raise ValueError(f"Run does not match cohort key: {run}")
    if metadata["cohort"] not in cohort or result["status"] != "PASS":
        raise ValueError(f"Unexpected cohort or failed run: {run}")
    if metadata["training_manifest_sha256"] != data.settings["training_manifest_sha256"]:
        raise ValueError(f"Training data version changed: {run}")
    if resolved["suite_revision"] != config["revision"] or resolved["selection"] != config["selection"]:
        raise ValueError(f"Selection rules or suite revision do not match: {run}")
    if not all(quality.get(key) for key in ("reload_scores", "reload_predictions",
                                           "validation_recomputed", "train_validation_hashes")):
        raise ValueError(f"Missing quality verification: {run}")
    if ids != [row.sample_id for row in data.validation] or not np.array_equal(
        y, data.encode(data.validation)
    ):
        raise ValueError(f"Prediction IDs or labels do not match validation: {run}")
    cutoff = threshold["threshold"]
    if not np.array_equal(y_pred, scores >= cutoff):
        raise ValueError(f"Threshold differs from recorded predictions: {run}")
    objective = float(f1_score(y, y_pred, average="macro", zero_division=0,
                               labels=[0, 1] if data.binary else None))
    if abs(objective - validation["macro_f1"]) > config["quality"]["metric_recompute_atol"]:
        raise ValueError(f"Independent Macro-F1 recomputation differs: {run}")
    if batch_metrics.measure(y, scores, cutoff, data.binary) != validation:
        raise ValueError(f"Validation metrics differ from prediction records: {run}")
    tuning = batch_metrics.select_threshold(y, scores, data.binary, config["selection"],
                                            threshold["score_kind"])
    if tuning != threshold or abs(result["macro_f1"] - objective) > 1e-12:
        raise ValueError(f"Validation threshold or candidate score differs: {run}")
    if experiment == "E5-FROZEN" and (
        not result["frozen_embedding_verified"] or result["initial_embedding_sha256"] !=
        result["best_embedding_sha256"] or result["initial_embedding_sha256"] !=
        result["final_embedding_sha256"]
    ):
        raise ValueError(f"Frozen embedding not verified: {run}")
    tracked = [_checkpoint(run, experiment), resolved_path, predictions, run / "thresholds.json",
               run / "metrics_validation.json", run / "run_manifest.json", run / "result.json",
               run / "quality.json"]
    return {"run": str(run), "seed": seed, "validation_macro_f1": objective,
            "files": {item.relative_to(run).as_posix(): sha256_file(item) for item in tracked}}


def _check_ablation(runs):
    configs = [load_yaml(Path(run) / "resolved_config.yaml")["settings"] for run in runs]
    summaries = [read_json(Path(run) / "result.json") for run in runs]
    for field in ("non_embedding_init_sha256", "vocabulary_sha256"):
        if len({summary[field] for summary in summaries}) != 1:
            raise ValueError(f"Unmatched E2/E5 {field}")
    if configs[1:] != configs[:1] * 2:
        raise ValueError("Unmatched E2/E5 hyperparameters")
    if summaries[1]["initial_embedding_sha256"] != summaries[2]["initial_embedding_sha256"]:
        raise ValueError("E5 conditions did not share the same pretrained matrix")


def _check_search(queue, config, dataset, experiment):
    name = f"{dataset}-{experiment}"
    study = queue["proposals"][name]
    accepted = [(trial["number"], trial["run"], trial["macro_f1"])
                for trial in study.values() if trial["status"] == "PASS"]
    initial = (config["hpo"]["per_dataset_svm_trials"] if experiment == "E1" else
               config["hpo"]["per_dataset_neural_model_trials"])
    required = 1 if experiment == "E1" else config["hpo"]["neural_min_accepted_trials"]
    if len(study) < initial or len(accepted) < required:
        raise ValueError(f"Incomplete Optuna study evidence: {name}")
    baseline = queue["tasks"][f"baseline/{dataset}/{experiment}/{config['seed']}"][-1]
    accepted.append((-1, baseline, read_json(Path(baseline) / "result.json")["macro_f1"]))
    best = max(value for _, _, value in accepted)
    winners = (item for item in accepted if best - item[2] <= config["selection"]["score_tie_atol"])
    if min(winners)[1] != queue["main"][f"{dataset}/{experiment}"]:
        raise ValueError(f"Recorded main result differs from validation ranking: {name}")


def _check_replication_keys(queue, config, datasets):
    required = set()
    spec = config["replication"]
    for dataset in datasets:
        recurrent = spec["select_best_of_per_dataset"]
        scores = {model: read_json(Path(queue["main"][f"{dataset}/{model}"]) /
                                   "result.json")["macro_f1"] for model in recurrent}
        best = max(scores.values())
        chosen = next(model for model in recurrent if best - scores[model] <=
                      config["selection"]["score_tie_atol"])
        for model in [*spec["required_experiments_per_dataset"], chosen]:
            required.update(f"{dataset}/{model}/{seed}" for seed in spec["seeds"])
    if set(queue["replications"]) != required or len(required) != 30:
        raise ValueError("Incomplete or duplicate three-seed replication keys")


def audit(config, datasets, output, progress):
    path = output / "final_runs.yaml"
    if path.exists():
        raise FileExistsError(f"Audit already frozen: {path}")
    queue_file = output / "queue.json"
    queue = read_json(queue_file)
    training = {key: load_training_data(item) for key, item in datasets.items()}
    experiments = [item["id"] for item in config["experiments"]]
    expected = {f"{dataset}/{model}" for dataset in training for model in experiments}
    if set(queue["main"]) != expected or len(expected) != 14:
        raise ValueError("Audit requires all 14 main results")
    _check_replication_keys(queue, config, datasets)
    main, replications, optional = {}, {}, {}
    seed = config["seed"]
    for dataset, data in training.items():
        checked_file(Path(data.settings["frozen_root"]) / "dataset_manifest.json",
                     data.settings["dataset_manifest_sha256"])
        for model in experiments:
            key = f"{dataset}/{model}"
            progress.emit("AUDIT", f"main {key}", verified=len(main), expected=14)
            main[key] = _validate_run(Path(queue["main"][key]), data, model,
                                      {"baseline", "hpo", "ablation"}, seed, config)
            if not model.startswith("E5"):
                _check_search(queue, config, dataset, model)
        _check_ablation([main[f"{dataset}/{model}"]["run"]
                        for model in ["E2", "E5-TUNE", "E5-FROZEN"]])
        for rep_seed in config["replication"]["seeds"]:
            _check_ablation([queue["replications"][f"{dataset}/{model}/{rep_seed}"]
                            for model in ["E2", "E5-TUNE", "E5-FROZEN"]])
    for store, role, group in ((replications, "replication", "replications"),
                                (optional, "extra_replication", "optional_replications")):
        for key, value in queue.get(group, {}).items():
            run = Path(value)
            dataset, model, rep_seed = key.split("/")
            if group == "optional_replications" and (
                read_json(run / "result.json")["status"] != "PASS"
            ):
                continue
            progress.emit("AUDIT", f"{role} {key}", verified=len(store))
            store[key] = _validate_run(run, training[dataset], model, {role}, int(rep_seed), config)
            primary = Path(main[f"{dataset}/{model}"]["run"])
            if load_yaml(run / "resolved_config.yaml") != load_yaml(primary / "resolved_config.yaml"):
                raise ValueError(f"Replication parameters differ from main result: {run}")
    all_paths = [entry["run"] for group in (main, replications, optional)
                 for entry in group.values()]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("A physical run was included twice in the freeze")
    frozen = {"suite_revision": config["revision"], "main": main, "replications": replications,
              "optional_replications": optional, "dataset_manifest_sha256": {
                  name: data["dataset_manifest_sha256"] for name, data in datasets.items()
              }, "queue_sha256": sha256_file(queue_file), "bootstrap": config["bootstrap"],
              "evaluator_sha256": {name: sha256_file(file)
                                   for name, file in _evaluator_sources().items()}}
    write_atomic_json(output / "audit.json", {"status": "PASS", "main": len(main),
                                              "replications": len(replications),
                                              "optional": len(optional)})
    temp = path.with_suffix(".pending")
    temp.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    temp.rename(path)
    progress.emit("FROZEN", f"{len(all_paths)} audited Runs", freeze=path, sha256=sha256_file(path))
    return {"freeze": str(path), "frozen_runs": len(all_paths)}


def verified_freeze(config, datasets, output):
    freeze = output / "final_runs.yaml"
    frozen = load_yaml(freeze)
    if frozen["suite_revision"] != config["revision"] or frozen["bootstrap"] != config["bootstrap"]:
        raise ValueError("Suite changed after the freeze")
    checked_file(output / "queue.json", frozen["queue_sha256"])
    if len(frozen["main"]) != 14 or len(frozen["replications"]) != 30:
        raise ValueError("Invalid final run grid")
    for name, expected in frozen["dataset_manifest_sha256"].items():
        item = datasets[name]
        if expected != item["dataset_manifest_sha256"]:
            raise ValueError("Frozen dataset configuration changed")
        checked_file(Path(item["frozen_root"]) / "dataset_manifest.json", expected)
    for name, file in _evaluator_sources().items():
        checked_file(file, frozen["evaluator_sha256"][name])
    for role in ("main", "replications", "optional_replications"):
        for entry in frozen[role].values():
            run = Path(entry["run"])
            for file, digest in entry["files"].items():
                checked_file(run / file, digest)
    return frozen, sha256_file(freeze)


def final_test(config, datasets, output, progress, device):
    frozen, digest = verified_freeze(config, datasets, output)
    data = {name: load_training_data(item) for name, item in datasets.items()}
    test = {}
    for name, loaded in data.items():
        root = Path(loaded.settings["frozen_root"])
        manifest = read_json(root / "dataset_manifest.json")
        test[name] = load_partition(root, manifest["splits"]["test"], "test")
    items = [(key, item) for role in ("main", "replications", "optional_replications")
             for key, item in frozen[role].items()]
    for index, (key, entry) in enumerate(items, 1):
        run = Path(entry["run"])
        destination = run / "test"
        if destination.exists():
            done = read_json(destination / "evaluation_manifest.json")
            if done["freeze_sha256"] != digest:
                raise ValueError(f"Existing test uses a different freeze: {run}")
            for file, checksum in done["files"].items():
                checked_file(destination / file, checksum)
            continue
        dataset, experiment, *_ = key.split("/")
        loaded, rows = data[dataset], test[dataset]
        threshold = read_json(run / "thresholds.json")["threshold"]
        progress.emit("TEST", f"{key} inference {index}/{len(items)}", run=run, samples=len(rows))
        source = _checkpoint(run, experiment)
        if experiment == "E1":
            scores = svm_predict(joblib.load(source), rows)
        else:
            from nlp_text_classification.training.neural_runner import predict_model

            scores = predict_model(source, loaded, load_yaml(run / "resolved_config.yaml"),
                                   rows, device)
        y = loaded.encode(rows)
        with tempfile.TemporaryDirectory(prefix=".test-", dir=run) as temporary:
            result = Path(temporary) / "test"
            result.mkdir()
            files = ["metrics_test.json", "predictions_test.jsonl"]
            write_atomic_json(result / files[0], batch_metrics.measure(y, scores, threshold,
                                                                      loaded.binary))
            batch_metrics.write_predictions(result / files[1], rows, y, scores, threshold)
            write_atomic_json(result / "evaluation_manifest.json", {
                "freeze_sha256": digest, "dataset": dataset, "run": run.name,
                "files": {name: sha256_file(result / name) for name in files},
            })
            result.rename(destination)
    progress.emit("TEST_COMPLETE", f"{len(items)} frozen evaluations published", freeze=digest)
    return {"evaluated": len(items), "freeze_sha256": digest}


def evaluation_command(mode, config, datasets, output, progress, device):
    with exclusive_queue(output.parent / ".batch-a-gpu.lock"):
        if mode == "audit":
            return audit(config, datasets, output, progress)
        if mode == "final-test":
            return final_test(config, datasets, output, progress, device)
        from nlp_text_classification.evaluation.batch_report import build_report

        return build_report(config, datasets, output, progress)
