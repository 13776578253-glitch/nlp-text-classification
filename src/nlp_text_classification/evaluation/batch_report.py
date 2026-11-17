"""Rebuild Batch A report artifacts from frozen full-precision predictions."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from nlp_text_classification.evaluation.batch_metrics import measure, read_predictions
from nlp_text_classification.experiments.final_evaluation import verified_freeze
from nlp_text_classification.experiments.suite import checked_file, load_yaml, read_json
from nlp_text_classification.training.progress import write_atomic_json


def _binary_macro(y, predictions):
    matrix = np.stack([y == 0, y == 1], axis=-1)
    guess = np.stack([predictions == 0, predictions == 1], axis=-1)
    common = (matrix & guess).sum(axis=1)
    total = matrix.sum(axis=1) + guess.sum(axis=1)
    return np.divide(2 * common, total, out=np.zeros_like(total, dtype=float),
                     where=total != 0).mean(axis=1)


def paired_macro_f1_interval(y, a, b, bootstrap):
    y, a, b = (np.asarray(arr).reshape(-1) for arr in (y, a, b))
    groups = [np.flatnonzero(y == value) for value in (0, 1)]
    random = np.random.default_rng(bootstrap["seed"])
    offset = 0
    result = np.empty(bootstrap["n_resamples"])
    while offset < len(result):
        chunk = min(500, len(result) - offset)
        indices = np.column_stack([
            random.choice(group, (chunk, len(group)), replace=True) for group in groups if len(group)
        ])
        labels = y[indices]
        result[offset:offset + chunk] = (
            _binary_macro(labels, a[indices]) - _binary_macro(labels, b[indices])
        )
        offset += chunk
    tail = (1 - bootstrap["confidence_level"]) / 2
    low, high = np.quantile(result, [tail, 1 - tail])
    return {"lower": float(low), "upper": float(high), "samples": len(y),
            "resamples": len(result), "seed": bootstrap["seed"]}


def _label_interval(y, predicted, settings):
    generator = np.random.default_rng(settings["seed"])
    samples = np.empty(settings["n_resamples"])
    for start in range(0, len(samples), 200):
        count = min(200, len(samples) - start)
        index = generator.integers(len(y), size=(count, len(y)))
        yt, pred = y[index], predicted[index]
        total = yt.sum(1) + pred.sum(1)
        samples[start:start + count] = np.divide(
            2 * (yt & pred).sum(1), total, out=np.zeros(count), where=total != 0
        )
    tail = (1 - settings["confidence_level"]) / 2
    return np.quantile(samples, [tail, 1 - tail]).tolist()


def _read_result(entry, dataset, freeze):
    run = Path(entry["run"])
    base = run / "test"
    audit = read_json(base / "evaluation_manifest.json")
    if audit["freeze_sha256"] != freeze:
        raise ValueError(f"Test result belongs to another freeze: {run}")
    for path, digest in audit["files"].items():
        checked_file(base / path, digest)
    cutoff = read_json(run / "thresholds.json")["threshold"]
    ids, y, scores, pred = read_predictions(base / "predictions_test.jsonl")
    if not np.array_equal(pred, scores >= cutoff):
        raise ValueError(f"Prediction decisions disagree with frozen threshold at {run}")
    actual = measure(y, scores, cutoff, dataset == "movie_reviews_zh")
    if read_json(base / "metrics_test.json") != actual:
        raise ValueError(f"Report could not reproduce official test metrics for {run}")
    return {"run": run, "validation": entry["validation_macro_f1"], "test": actual,
            "ids": ids, "y": y, "predictions": pred}


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def _curve(run, target, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (run / "history.csv").open(newline="", encoding="utf-8") as stream:
        points = list(csv.DictReader(stream))
    fig, scores = plt.subplots(figsize=(6, 3))
    x = [int(row["epoch"]) for row in points]
    scores.plot(x, [float(row["val_macro_f1"]) for row in points], label="Validation Macro-F1")
    if points and "loss" in points[0]:
        loss = scores.twinx()
        loss.plot(x, [float(row["loss"]) for row in points], color="darkorange",
                  label="Training loss")
        loss.set_ylabel("Training loss")
    scores.set_xlabel("Epoch")
    scores.set_ylabel("Validation Macro-F1")
    scores.set_title(label)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)


def _append_trials(lines, queue):
    lines.extend(["## Hyperparameter search", ""])
    rows = []
    for study, proposals in queue["proposals"].items():
        for choice in proposals.values():
            result = (f"{choice['macro_f1']:.4f}" if "macro_f1" in choice else "—")
            rows.append((study, choice["number"], choice["status"], result,
                         f"{choice.get('elapsed_seconds', 0):.2f}", choice["reason"],
                         Path(choice["run"]).name, "`" + str(choice["params"]) + "`"))
    lines.extend(_table(["Study", "Trial", "Status", "Validation Macro-F1", "Seconds", "Stop reason",
                         "Run", "Parameters"], rows))


def _append_replications(lines, frozen, digest):
    lines.extend(["## Independent seed replications", ""])
    groups = defaultdict(dict)
    for key, entry in frozen["replications"].items():
        dataset, experiment, seed = key.split("/")
        result = _read_result(entry, dataset, digest)
        group = groups[(dataset, experiment)]
        if seed in group:
            raise ValueError(f"Repeated independent seed: {key}")
        group[seed] = (result["validation"], result["test"]["macro_f1"])
    rows = []
    for (dataset, experiment), seeds in sorted(groups.items()):
        if set(seeds) != {"42", "123", "2026"}:
            raise ValueError(f"Missing mandatory seed for {dataset}/{experiment}")
        values = np.array(list(seeds.values()))
        val_mean, test_mean = values.mean(0)
        val_sd, test_sd = values.std(0, ddof=1)
        individual = ", ".join(f"{seed}: {seeds[seed][1]:.4f}" for seed in ("42", "123", "2026"))
        rows.append((dataset, experiment, individual, len(seeds), f"{val_mean:.4f} ± {val_sd:.4f}",
                     f"{test_mean:.4f} ± {test_sd:.4f}"))
    lines.extend(_table(["Dataset", "Model", "Test Macro-F1 by seed", "n",
                         "Validation mean ± SD", "Test mean ± SD"], rows))
    if frozen["optional_replications"]:
        optional = []
        for key, entry in frozen["optional_replications"].items():
            dataset = key.split("/")[0]
            result = _read_result(entry, dataset, digest)
            optional.append((key, result["run"].name, f"{result['test']['macro_f1']:.4f}"))
        lines.extend(["### Additional validation-selected runs", ""])
        lines.extend(_table(["Run key", "Run ID", "Test Macro-F1"], optional))
    return rows


def _append_intervals(lines, models, config, dataset_specs):
    bootstrap = config["bootstrap"]
    chinese = models["movie_reviews_zh"]
    order = config["replication"]["select_best_of_per_dataset"]
    best = max(chinese[model]["validation"] for model in order)
    best_recurrent = next(model for model in order if best - chinese[model]["validation"] <=
                          config["selection"]["score_tie_atol"])
    intervals = []
    for left, right in bootstrap["movie_reviews_zh"]["macro_f1_contrasts"]:
        a = chinese[best_recurrent if left == "best_recurrent" else left]
        b = chinese[right]
        if a["ids"] != b["ids"] or not np.array_equal(a["y"], b["y"]):
            raise ValueError("Paired bootstrap requires aligned test IDs and labels")
        interval = paired_macro_f1_interval(a["y"], a["predictions"], b["predictions"], bootstrap)
        diff = a["test"]["macro_f1"] - b["test"]["macro_f1"]
        intervals.append((best_recurrent if left == "best_recurrent" else left, right,
                          f"{diff:.4f}", f"[{interval['lower']:.4f}, {interval['upper']:.4f}]",
                          interval["samples"]))
    lines.extend(["## Chinese paired Bootstrap", "",
                  f"{bootstrap['n_resamples']:,} label-stratified paired resamples.", ""])
    lines.extend(_table(["Model A", "Model B", "Macro-F1 A − B", "95% percentile CI", "n"], intervals))
    root = Path(dataset_specs["goemotions"]["frozen_root"])
    manifest = read_json(root / "dataset_manifest.json")
    labels = manifest["label_names"]
    support = manifest["derived_summary"]["label_counts"]["train"]
    conditions = bootstrap["goemotions"]
    rare = []
    low_support = sorted(range(len(labels)), key=lambda index: support[str(index)])[:5]
    for experiment, item in models["goemotions"].items():
        for index in sorted(set(low_support) | {i for i in range(len(labels)) if (
            support[str(i)] < conditions["rare_label_train_support_below"]
        )}):
            test_support = item["test"]["per_label_support"][index]
            ci = "—"
            if (support[str(index)] < conditions["rare_label_train_support_below"] and
                test_support >= conditions["min_positive_test_support_for_ci"]):
                result = _label_interval(item["y"][:, index].astype(bool),
                                         item["predictions"][:, index].astype(bool), bootstrap)
                ci = f"[{result[0]:.4f}, {result[1]:.4f}]"
            rare.append((experiment, labels[index], support[str(index)], test_support,
                         f"{item['test']['per_label_f1'][index]:.4f}", ci))
    lines.extend(["## GoEmotions low-support labels", "",
                  "Five lowest train support labels and all labels meeting the rare-label rule.", ""])
    lines.extend(_table(["Model", "Label", "Train support", "Test support", "F1", "95% CI"], rare))
    return intervals


def _write_latex(path, models, rows, intervals, freeze):
    endrow = r" \\"
    lines = [r"\documentclass{article}", r"\usepackage[margin=24mm]{geometry}",
             r"\begin{document}", r"\title{EmotionLab: Batch A V2}", r"\maketitle",
             r"\section{Protocol}", ("The two datasets were audited and treated as separate tasks. "
             "Validation Macro-F1 governed model, checkpoint, and threshold selection. "
             "The test sets were evaluated after run identity and model artifacts were frozen."),
             r"\subsection{Traceability}", r"Freeze: \texttt{" + freeze[:16] + "}.", ""]
    for name, items in models.items():
        lines.extend([r"\section{" + name.replace("_", r"\_") + "}", "",
                      r"\begin{tabular}{lrr}",
                      "Model & Validation Macro-F1 & Test Macro-F1" + endrow, r"\hline"])
        for model, row in items.items():
            lines.append(f"{model} & {row['validation']:.4f} & {row['test']['macro_f1']:.4f}" + endrow)
        lines.extend([r"\end{tabular}", ""])
    lines.extend([r"\section{Embedding ablation}",
                  ("The random condition reuses E2. Both pretrained conditions use identical "
                  "non-embedding initialization and E2 hyperparameters. The frozen condition "
                  "was verified by comparing the embedding bytes before and after training."),
                  r"\section{Independent replications}", r"\begin{tabular}{llrr}",
                  "Dataset & Model & n & Test Macro-F1 (mean and SD)" + endrow, r"\hline"])
    for dataset, experiment, _, count, _, test in rows:
        test = test.replace("±", r"$\pm$")
        lines.append(" & ".join([dataset.replace("_", r"\_"), experiment, str(count), test]) + endrow)
    lines.extend([r"\end{tabular}", r"\section{Paired Bootstrap on Chinese reviews}",
                  r"\begin{tabular}{llrr}", r"A & B & Difference & 95\% CI" + endrow, r"\hline"])
    for left, right, diff, ci, _ in intervals:
        lines.append(f"{left} & {right} & {diff} & {ci}" + endrow)
    lines.extend([r"\end{tabular}", "", ("See the generated Markdown companion for trial parameters, "
                  "per-class metrics, convergence plots, and evidence paths."), r"\end{document}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(config, datasets, output, progress):
    frozen, digest = verified_freeze(config, datasets, output)
    destination = Path("reports") / "batch-a" / digest[:12]
    destination.mkdir(parents=True, exist_ok=True)
    curves = destination / "curves"
    curves.mkdir(exist_ok=True)
    lines = ["# EmotionLab Batch A results", "", f"Frozen manifest SHA-256: `{digest}`", "",
             ("Each main row uses the seed 42 selected Run. "
             "The independent seed cohort is reported separately."), ""]
    models = defaultdict(dict)
    all_results = {}
    for key, entry in frozen["main"].items():
        dataset, model = key.split("/")
        result = _read_result(entry, dataset, digest)
        models[dataset][model] = result
        all_results[key] = {"run": str(result["run"]), **result["test"]}
    for dataset, items in models.items():
        lines.extend([f"## {dataset}", ""])
        rows = []
        for model, item in items.items():
            run = item["run"]
            tested = item["test"]
            params = load_yaml(run / "resolved_config.yaml")["settings"]
            result = read_json(run / "result.json")
            entry = (model, f"{item['validation']:.4f}", f"{tested['macro_f1']:.4f}",
                     f"{tested['micro_f1']:.4f}", result.get("best_epoch", "—"),
                     result["stop_reason"], run.name, "`" + str(params) + "`")
            rows.append(entry)
            _curve(run, curves / f"{dataset}-{model}.png", f"{dataset} · {model}")
        lines.extend(_table(["Model", "Validation Macro-F1", "Test Macro-F1", "Test Micro-F1",
                             "Best epoch", "Stop reason", "Run ID", "Selected parameters"], rows))
        lines.extend(["### Embedding ablation", ""])
        ablation = []
        for model in ("E2", "E5-TUNE", "E5-FROZEN"):
            item = items[model]
            result = read_json(item["run"] / "result.json")
            vectors = result["word2vec"]
            ablation.append((model, f"{item['test']['macro_f1']:.4f}",
                             vectors.get("vocabulary_coverage", "—"),
                             vectors.get("token_coverage", "—"), item["run"].name))
        lines.extend(_table(["Model", "Test Macro-F1", "Vocabulary coverage", "Token coverage",
                             "Run ID"], ablation))
        all_results[f"{dataset}/E5-RANDOM"] = {"references": f"{dataset}/E2"}
    queue = read_json(output / "queue.json")
    _append_trials(lines, queue)
    replications = _append_replications(lines, frozen, digest)
    intervals = _append_intervals(lines, models, config, datasets)
    lines.extend(["## Failed, pruned, or interrupted attempts", ""])
    failures = []
    for key, attempts in queue["tasks"].items():
        for path in attempts:
            result_path = Path(path) / "result.json"
            item = read_json(result_path) if result_path.is_file() else {
                "status": "INCOMPLETE", "reason": "Missing terminal event"
            }
            if item["status"] != "PASS":
                failures.append((key, path, item["status"], item["reason"].replace("|", "\\|")))
    lines.extend(_table(["Task", "Run", "Status", "Reason"], failures))
    review = read_json(output / "PERFORMANCE_REVIEW.json")
    lines.extend(["## Performance review", ""])
    lines.extend(_table(["Dataset", "Model", "Flags", "Train Macro-F1", "Validation Macro-F1"],
                         [(item["dataset"], item["experiment"], ", ".join(item["flags"]),
                           f"{item['train_macro_f1']:.4f}", f"{item['validation_macro_f1']:.4f}")
                          for item in review["reviewed"]]))
    lines.extend(["## Reproduction", "",
                  "Configure the two frozen dataset roots and the approved suite revision.",
                  ("Run `batch-a smoke --output artifacts/batch-a-smoke`, followed by "
                  "`batch-a run`, `batch-a audit`, `batch-a final-test`, and `batch-a report` "
                  "with the same Batch A artifact location."),
                  "", "Per-label metrics and confusion matrices: [class_metrics.json](class_metrics.json).",
                  "", f"Independent audit evidence: `{output / 'audit.json'}`.", ""])
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_atomic_json(destination / "class_metrics.json", all_results)
    write_atomic_json(destination / "quality_review.json", {
        "audit": read_json(output / "audit.json"), "performance_review": review,
        "failures": failures, "frozen_manifest_sha256": digest,
    })
    _write_latex(destination / "report.tex", models, replications, intervals, digest)
    progress.emit("REPORT_COMPLETE", "Recomputed frozen predictions and generated reports",
                  report=destination / "report.md", latex=destination / "report.tex")
    return {"report": str(destination / "report.md"), "latex": str(destination / "report.tex")}
