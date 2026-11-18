"""Publish three-seed validation evidence, label sensitivity, and a freeze proposal."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, stdev

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.evaluation.batch_a_s1_report import _table
from nlp_text_classification.evaluation.batch_b_report import _history
from nlp_text_classification.experiments.batch_b_resources import label_names
from nlp_text_classification.experiments.batch_b_s2 import OUTPUT, PROBE
from nlp_text_classification.experiments.suite import load_yaml, read_json
from nlp_text_classification.training.progress import write_atomic_json

REPORT = Path("reports/batch-b-s2")
METRICS = ("macro_f1", "micro_f1", "subset_accuracy", "accuracy")


def rare_gap(left, right, rare):
    """Use all-label weighting for additive contributions and keep a separate sensitivity score."""
    width = len(left)
    rest = [index for index in range(width) if index not in rare]
    differences = [a - b for a, b in zip(left, right, strict=True)]
    return {
        "all_label_gap": math.fsum(differences) / width,
        "rare_contribution": math.fsum(differences[index] for index in rare) / width,
        "remaining_contribution": math.fsum(differences[index] for index in rest) / width,
        "remaining_label_gap": mean(differences[index] for index in rest),
        "per_label_gap": differences,
    }


def _status(entry, stage):
    run = Path(entry["run"])
    resolved = load_yaml(run / "resolved_config.yaml")
    result = entry["result"]
    metrics = read_json(run / "metrics_validation.json")
    quality = read_json(run / "quality.json")
    return {
        "stage": stage, "run": str(run), "checkpoint": result["checkpoint"],
        "files": entry["files"], "manifest": read_json(run / "run_manifest.json"),
        "settings": resolved["settings"], "dataset": resolved["experiment"]["dataset"],
        "model_id": resolved["experiment"]["model_id"],
        "model_revision": resolved["resource"]["revision"],
        "tokenization": resolved["tokenization"],
        "metric_protocol": resolved["selection"],
        "validation": metrics, "threshold": result["threshold"], "score_kind": quality["score_kind"],
        "best_epoch": result["best_epoch"], "epochs_completed": result["epochs_completed"],
        "stop_reason": result["stop_reason"], "elapsed_seconds": result["elapsed_seconds"],
        "peak_reserved_bytes": result["peak_reserved_bytes"],
        "planned_steps": quality["planned_steps"], "warmup_steps": quality["warmup_steps"],
    }


def _cohort(inputs, queue, decision):
    probe = _status(queue["accepted"][PROBE], "S2")
    original = {key: _status(run, "S1") for key, run in inputs.originals.items()}
    long = decision["selected_budget"] == "s2_12_epochs"
    cohort = {}
    for key in inputs.models:
        cohort[f"{key}/42"] = probe if key == "B1" and long else original[key]
        for seed in inputs.seeds[1:]:
            slot = f"replication/{key}/{seed}"
            cohort[f"{key}/{seed}"] = _status(queue["accepted"][slot], "S2")
    return cohort, {"short": original["B1"], "long": probe}


def _aggregate(cohort, keys, seeds):
    results = {}
    for model in keys:
        runs = {str(seed): cohort[f"{model}/{seed}"]["validation"] for seed in seeds}
        result = {}
        for field in METRICS:
            if all(field in row for row in runs.values()):
                scores = {seed: row[field] for seed, row in runs.items()}
                result[field] = {"seed_values": scores, "mean": mean(scores.values()),
                                 "sample_std": stdev(scores.values()), "n": len(scores)}
        results[model] = result
    return results


def _labels(inputs, cohort):
    labels = {}
    for dataset, data in inputs.data.items():
        names = label_names(data)
        train = data.encode(data.train).sum(axis=0).tolist()
        validation = data.encode(data.validation).sum(axis=0).tolist()
        if data.binary:
            train.insert(0, len(data.train) - train[0])
            validation.insert(0, len(data.validation) - validation[0])
        labels[dataset] = {
            "names": names, "train_support": train, "validation_support": validation, "runs": {},
        }
    for key, entry in cohort.items():
        block = labels[entry["dataset"]]
        metrics = entry["validation"]
        if metrics["per_label_support"] != block["validation_support"]:
            raise ValueError(f"Per-label supports differ from the frozen partition: {key}")
        block["runs"][key] = {field: metrics[f"per_label_{field}"]
                              for field in ("precision", "recall", "f1", "support")}
    return labels


def _ranking(scores):
    groups = []
    for model in sorted(scores, key=scores.get, reverse=True):
        if groups and abs(scores[groups[-1][0]] - scores[model]) <= 1e-12:
            groups[-1].append(model)
        else:
            groups.append([model])
    return groups


def _diagnostics(inputs, labels, cohort):
    english = labels["goemotions"]
    width = len(english["names"])
    rare = sorted(range(width), key=lambda index: (english["train_support"][index], index))[
        :inputs.spec["rare_label_count"]
    ]
    common = [index for index in range(width) if index not in rare]
    comparisons, rankings = {}, {}
    keys = tuple(key for key, item in inputs.models.items() if item["dataset"] == "goemotions")
    for seed in inputs.seeds:
        vectors = {model: english["runs"][f"{model}/{seed}"]["f1"] for model in keys}
        scores = {key: cohort[f"{key}/{seed}"]["validation"] for key in keys}
        comparison = rare_gap(vectors["B2"], vectors["B3"], rare)
        if abs(comparison["all_label_gap"] - (scores["B2"]["macro_f1"]
                                             - scores["B3"]["macro_f1"])) > 1e-12:
            raise ValueError("Label gap does not reconstruct the accepted Macro-F1 gap")
        comparisons[str(seed)] = comparison
        rankings[str(seed)] = {
            field: _ranking({key: scores[key][field] for key in keys})
            for field in ("macro_f1", "micro_f1", "subset_accuracy")
        }
        rankings[str(seed)]["macro_f1_without_low_support_labels"] = _ranking({
            key: mean(vectors[key][index] for index in common) for key in keys
        })
    rare_reversals = [seed for seed, gap in comparisons.items()
                      if gap["all_label_gap"] * gap["remaining_label_gap"] < 0]
    return {"rare_label_indices": rare, "rare_label_names": [english["names"][i] for i in rare],
            "macro_f1_gap_b2_minus_b3": comparisons, "rankings": rankings,
            "rare_sensitive_reversal_seeds": rare_reversals}


def _plot_budget(budget):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 1, figsize=(7.3, 4.7), sharex=True, layout="constrained")
    for label, entry in budget.items():
        rows = _history(entry["run"])
        epochs = [int(row["epoch"]) for row in rows]
        color = "tab:blue" if label == "short" else "tab:orange"
        for axis, field in zip(axes, ("val_macro_f1", "learning_rate"), strict=True):
            axis.plot(epochs, [float(row[field]) for row in rows], marker="o",
                      color=color, label=f"{entry['settings']['epochs']}-epoch schedule")
        axes[0].scatter(entry["best_epoch"], entry["validation"]["macro_f1"], marker="*",
                        s=135, color=color, zorder=4)
    axes[0].set_title("B1 DistilBERT · seed 42 · independent schedules")
    axes[0].set_ylabel("Validation Macro-F1")
    axes[1].set_ylabel("Epoch-end learning rate")
    axes[1].set_xlabel("Epoch")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    fig.savefig(REPORT / "b1-budget.png", dpi=160)
    plt.close(fig)


def _freeze(inputs, cohort, decision):
    rows = []
    for key, run in cohort.items():
        model, seed = key.split("/")
        dataset = run["dataset"]
        rows.append({
            "model": model, "seed": int(seed), "role": "primary" if seed == "42" else "replication",
            "stage": run["stage"], "run": run["run"], "checkpoint": run["checkpoint"],
            "dataset": dataset, "training_manifest_sha256": run["manifest"]["training_manifest_sha256"],
            "dataset_manifest_sha256": inputs.datasets[dataset]["dataset_manifest_sha256"],
            "label_order": label_names(inputs.data[dataset]),
            "positive_raw_id": inputs.datasets[dataset].get("positive_raw_id"),
            "model_id": run["model_id"], "model_revision": run["model_revision"],
            "tokenizer_sha256": run["tokenization"]["tokenizer_sha256"],
            "settings": run["settings"], "tokenization": run["tokenization"],
            "selection": run["metric_protocol"], "best_epoch": run["best_epoch"],
            "validation_macro_f1": run["validation"]["macro_f1"], "threshold": run["threshold"],
            "score_kind": run["score_kind"], "files": run["files"],
        })
    return {"status": "proposed", "suite": inputs.identity, "source_split": "validation",
            "primary_seed": inputs.seeds[0], "replication_seeds": inputs.seeds[1:],
            "b1_budget_decision": decision, "runs": rows}


def _score_cell(metrics, key):
    return f"{metrics[key]:.4f}" if key in metrics else "—"


def _write_report(cohort, summary, budget, decision, label_results, diagnostics):
    lines = [
        "# EmotionLab · Batch B S2 三种子验证", "",
        ("B1–B4 使用冻结 train/validation 分区和各自 S1 入选超参数。"
         "B2、B3、B4 的 seed 42 来自 S1。B1 根据预算对照选择同协议的三种子结果。"
         "模型比较按数据集进行。"), "",
        "## B1 训练预算", "",
    ]
    rows = []
    for key, run in budget.items():
        rows.append([key, str(run["settings"]["epochs"]), str(run["settings"]["patience"]),
                     f'{run["planned_steps"]:,}', str(run["warmup_steps"]),
                     f'{run["best_epoch"]}/{run["epochs_completed"]}',
                     f'{run["validation"]["macro_f1"]:.4f}', run["manifest"]["run_id"]])
    lines.extend(_table(["协议", "Epoch 上限", "Patience", "计划更新", "Warmup",
                         "最佳/实际轮次", "验证 Macro-F1", "Run"], rows))
    lines.extend(["", (f'12 轮协议的验证增益为 **{decision["gain"]:+.4f}**，'
                       f'采用门槛为 **{decision["minimum_macro_f1_gain"]:.4f}**。'
                       f'入选预算为 `{decision["selected_budget"]}`，两个追加种子沿用该预算。'),
                  "", "[Seed 42 验证曲线与 Linear Scheduler](b1-budget.png)",
                  "", "## 三种子汇总", ""])
    rows = []
    for key, entries in summary.items():
        for metric, score in entries.items():
            values = [f"{item:.4f}" for item in score["seed_values"].values()]
            rows.append([key, metric, *values, f'{score["mean"]:.4f}',
                         f'{score["sample_std"]:.4f}', str(score["n"])])
    lines.extend(_table(["模型", "验证指标", "Seed 42", "Seed 123", "Seed 2026",
                         "均值", "样本标准差", "n"], rows))
    lines.extend(["", "## 逐种子结果与英文排名", ""])
    rows = []
    for key, run in cohort.items():
        model, seed = key.split("/")
        metrics = run["validation"]
        rows.append([
            model, seed, run["stage"], _score_cell(metrics, "macro_f1"),
            _score_cell(metrics, "micro_f1"), _score_cell(metrics, "subset_accuracy"),
            _score_cell(metrics, "accuracy"), f'{run["best_epoch"]}/{run["epochs_completed"]}',
            f'{run["threshold"]:.2f}', run["manifest"]["run_id"],
        ])
    lines.extend(_table(["模型", "Seed", "来源", "Macro-F1", "Micro-F1", "Subset Accuracy",
                         "Accuracy", "最佳/实际轮次", "阈值", "Run ID"], rows))
    lines.append("")
    rows = [[seed, *(" > ".join(" = ".join(group) for group in value) for value in ranks.values())]
            for seed, ranks in diagnostics["rankings"].items()]
    lines.extend(_table(["Seed", "Macro-F1 排名", "Micro-F1 排名", "Subset Accuracy 排名",
                         "去掉最低支持组后的 Macro-F1 排名"], rows))
    lines.extend(["", "## 低支持标签对 B2、B3 差值的贡献", "",
                  ("按训练正例支持量取最低五类。每个标签的 F1 差以 1/28 权重计入"
                   "原 Macro-F1 差。去掉该组重算的分数使用各 Run 的原阈值，仅用于敏感性分析。"),
                  ""])
    english = label_results["goemotions"]
    rows = []
    for index in diagnostics["rare_label_indices"]:
        values = [f'{row["per_label_gap"][index]:+.4f}'
                  for row in diagnostics["macro_f1_gap_b2_minus_b3"].values()]
        rows.append([english["names"][index], str(english["train_support"][index]),
                     str(english["validation_support"][index]), *values])
    lines.extend(_table(["标签", "训练正例", "验证正例", "ΔF1 seed 42", "ΔF1 seed 123",
                         "ΔF1 seed 2026"], rows))
    lines.append("")
    rows = [[seed, *(f'{row[field]:+.4f}' for field in ("all_label_gap", "rare_contribution",
                                                       "remaining_contribution",
                                                       "remaining_label_gap"))]
            for seed, row in diagnostics["macro_f1_gap_b2_minus_b3"].items()]
    lines.extend(_table(["Seed", "B2−B3 Macro-F1", "五类贡献", "其余 23 类贡献",
                         "仅其余 23 类的 Macro-F1 差"], rows))
    comparisons = list(diagnostics["macro_f1_gap_b2_minus_b3"].values())
    rare_mean = mean(item["rare_contribution"] for item in comparisons)
    other_mean = mean(item["remaining_contribution"] for item in comparisons)
    reversals = diagnostics["rare_sensitive_reversal_seeds"]
    reversal_text = (
        "去掉最低支持组后，B2 和 B3 的相对顺序在 seed " + "、".join(reversals) + " 互换。"
        if reversals else "三个种子去掉最低支持组后均保持 B2 和 B3 的相对顺序。"
    )
    lines.extend(["", (f"三个种子的 B2−B3 平均差中，最低支持组贡献 {rare_mean:+.4f}，"
                       f"其余标签贡献 {other_mean:+.4f}。{reversal_text}")])
    lines.extend(["", "[逐标签 F1、精确率、召回率与支持量](label-metrics.json)", "",
                  "## 拟冻结清单", "",
                  ("四个主 Run 固定使用 seed 42，各保留 seed 123、2026 作为复验。"
                   "清单绑定数据、Tokenizer、标签、权重、配置和各自验证阈值。"
                   "正式测试须在独立冻结审计之后执行。"),
                  "", "[拟冻结清单](proposed-freeze.json) · [机器可读结果](results.json)", ""])
    (REPORT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def publish(inputs, queue, decision):
    REPORT.mkdir(parents=True, exist_ok=True)
    cohort, budget = _cohort(inputs, queue, decision)
    summary = _aggregate(cohort, inputs.models, inputs.seeds)
    labels = _labels(inputs, cohort)
    diagnostics = _diagnostics(inputs, labels, cohort)
    _plot_budget(budget)
    write_atomic_json(REPORT / "results.json", {
        "suite": inputs.identity, "validation_only": True, "b1_budget_decision": decision,
        "b1_budget_runs": budget, "summary": summary, "cohort": cohort,
        "label_diagnostics": diagnostics, "new_attempts": queue["attempts"],
    })
    write_atomic_json(REPORT / "label-metrics.json", labels)
    proposal = REPORT / "proposed-freeze.json"
    write_atomic_json(proposal, _freeze(inputs, cohort, decision))
    _write_report(cohort, summary, budget, decision, labels, diagnostics)
    artifacts = ("report.md", "results.json", "label-metrics.json", "proposed-freeze.json",
                 "b1-budget.png")
    write_atomic_json(OUTPUT / "delivery-audit.json", {
        "status": "PASS", "source_split": "validation", "new_accepted_runs": len(queue["accepted"]),
        "comparison_runs": len(cohort), "s1_runs_preserved": len(inputs.s1["runs"]),
        "queue_sha256": sha256_file(OUTPUT / "queue.json"), "freeze_proposal_status": "proposed",
        "reports": {file: sha256_file(REPORT / file) for file in artifacts},
    })
    return {"report": str(REPORT / "report.md"), "proposal": str(proposal),
            "b1_budget": decision["selected_budget"]}
