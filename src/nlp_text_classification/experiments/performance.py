"""Diagnose weak but technically accepted models using validation evidence."""

from pathlib import Path

import numpy as np

from nlp_text_classification.experiments.suite import load_yaml, read_json
from nlp_text_classification.training.progress import write_atomic_json


def flags(training, validation, sd, limits):
    found = []
    if (validation["modal_prediction_share"] >= limits["constant_prediction_fraction"] or (
        validation["predicted_label_fraction"] < limits["min_predicted_label_fraction"]
        and validation["macro_f1"] < limits["min_train_macro_f1"]
    )):
        found.append("degenerate_predictions")
    if max(training["macro_f1"], validation["macro_f1"]) < limits["min_train_macro_f1"]:
        found.append("underfitting")
    if sd is not None and sd >= limits["unstable_macro_f1_sd"]:
        found.append("unstable_across_seeds")
    return found


def review(batch):
    config = batch.config
    spec = config["performance_review"]
    priorities = ("degenerate_predictions", "underfitting", "unstable_across_seeds")
    report = {"reviewed": [], "diagnostics": []}
    for dataset in batch.data:
        flagged = []
        for key, path in batch.state["main"].items():
            if not key.startswith(dataset + "/"):
                continue
            run = Path(path)
            experiment = key.split("/", 1)[1]
            sample = [read_json(Path(rep) / "result.json")["macro_f1"]
                      for label, rep in batch.state["replications"].items()
                      if label.startswith(key + "/")]
            sd = float(np.std(sample, ddof=1)) if len(sample) >= 2 else None
            train = read_json(run / "metrics_train.json")
            validation = read_json(run / "metrics_validation.json")
            issues = flags(train, validation, sd, spec)
            item = {"dataset": dataset, "experiment": experiment, "run": str(run),
                    "flags": issues, "train_macro_f1": train["macro_f1"],
                    "validation_macro_f1": validation["macro_f1"],
                    "prediction_coverage": validation["predicted_label_fraction"],
                    "modal_prediction_share": validation["modal_prediction_share"], "sd": sd,
                    "validation_support": validation["per_label_support"],
                    "stop_reason": read_json(run / "result.json")["stop_reason"],
                    "history": str(run / "history.csv")}
            report["reviewed"].append(item)
            if issues:
                flagged.append(item)
        if flagged:
            flagged.sort(key=lambda item: priorities.index(item["flags"][0]))
            for item in flagged[:spec["max_extra_runs_per_dataset"]]:
                experiment = item["experiment"]
                params = load_yaml(Path(item["run"]) / "resolved_config.yaml")["settings"]
                seed = config["seed"]
                if item["flags"][0] == "unstable_across_seeds":
                    seed = 1729
                elif experiment == "E1":
                    known = {proposal["params"]["c"] for proposal in batch.state["proposals"][
                        f"{dataset}-E1"
                    ].values()}
                    known.add(config["svm"]["c"])
                    candidates = [float(c) for c in np.geomspace(0.01, 10.0, 31)
                                  if not any(np.isclose(c, current) for current in known)]
                    params["c"] = min(candidates, key=lambda c: abs(np.log(c / params["c"])))
                else:
                    params["epoch_cap"] = spec["diagnostic_epoch_cap"]
                    params["patience"] = params["epoch_cap"]
                run, result = batch.attempt(dataset, experiment, "performance_review", seed, params)
                report["diagnostics"].append({"reference": item["run"], "run": str(run), **result})
    write_atomic_json(batch.output / "PERFORMANCE_REVIEW.json", report)
