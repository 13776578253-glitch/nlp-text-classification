"""Recompute the immutable Batch B test results and paired comparisons."""

from pathlib import Path
from statistics import mean, stdev

import numpy as np

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.experiments.batch_b_final import (
    MODELS,
    OUTPUT,
    SEEDS,
    json_bytes,
    publish_once,
    read_test,
    run_key,
    test_inputs,
)
from nlp_text_classification.experiments.suite import read_json

REPORTS = Path("reports/batch-b-final")
ENGLISH = ("macro_f1", "micro_f1", "subset_accuracy")
CHINESE = ("macro_f1", "accuracy", "positive_precision", "positive_recall",
           "positive_f1", "roc_auc", "pr_auc")


def _features(truth, pred):
    return np.column_stack((2 * (truth & pred), truth.astype(np.int8) + pred,
                            (truth == pred).all(axis=1)))


def _measures(counts, sample_count):
    tp, denom = np.split(counts[..., :-1], 2, axis=-1)
    per_label = np.divide(tp, denom, out=np.zeros(tp.shape), where=denom > 0)
    numerator, denominator = tp.sum(axis=-1), denom.sum(axis=-1)
    micro = np.divide(numerator, denominator, out=np.zeros(numerator.shape),
                      where=denominator > 0)
    return np.stack((per_label.mean(axis=-1), micro, counts[..., -1] / sample_count), axis=-1)


def paired_bootstrap(truth, pairs, config):
    """Use the same resampled test rows for all fixed seeds and both models."""
    if tuple(config["metrics"]) != ENGLISH or config["models"] != ["B3", "B2"] or (
        config["unit"] != "aligned_test_row" or not config["reuse_rows_across_fixed_seeds"]
    ):
        raise ValueError("Unexpected frozen comparison protocol")
    total, width = truth.shape
    features = np.column_stack([
        _features(truth, pred) for pair in pairs for pred in pair
    ]).astype(np.float64)
    shape = (len(pairs), 2, 2 * width + 1)
    points = _measures(features.sum(axis=0).reshape(shape), total)
    n = config["n_resamples"]
    rng = np.random.default_rng(config["seed"])
    probability = np.full(total, 1.0 / total)
    draws = np.empty((n, len(pairs), len(ENGLISH)))
    for start in range(0, n, 128):
        count = min(128, n - start)
        weights = rng.multinomial(total, probability, size=count)
        resampled = _measures((weights @ features).reshape(count, *shape), total)
        draws[start:start + count] = resampled[:, :, 0] - resampled[:, :, 1]
    return points, draws


def _paired(truth, predictions, rows, config):
    pairs = [(predictions[f"B3-{seed}"], predictions[f"B2-{seed}"]) for seed in SEEDS]
    points, draws = paired_bootstrap(truth, pairs, config)
    for index, seed in enumerate(SEEDS):
        expected = [[rows[f"{model}-{seed}"]["test"][metric] for metric in ENGLISH]
                    for model in config["models"]]
        if not np.allclose(points[index], expected, atol=1e-12, rtol=0):
            raise ValueError(f"Paired point estimates differ from recomputed metrics: seed {seed}")
    tail = (1 - config["confidence"]) / 2

    def intervals(point, distribution):
        bounds = np.quantile(distribution, [tail, 1 - tail], axis=0)
        return {metric: {"difference": float(point[index]), "confidence_interval": [
            float(bounds[0, index]), float(bounds[1, index]),
        ]} for index, metric in enumerate(ENGLISH)}

    gap = points[:, 0] - points[:, 1]
    return {
        "protocol": config, "test_rows": len(truth),
        "by_seed": {str(seed): intervals(gap[index], draws[:, index])
                    for index, seed in enumerate(SEEDS)},
        "mean_of_fixed_seeds": intervals(gap.mean(axis=0), draws.mean(axis=1)),
    }


def _summary(rows):
    result = {}
    for model in MODELS:
        group = [rows[f"{model}-{seed}"] for seed in SEEDS]
        metrics = CHINESE if model == "B4" else ENGLISH
        values = {"validation_macro_f1": [item["validation_macro_f1"] for item in group]}
        values.update({key: [item["test"][key] for item in group] for key in metrics})
        result[model] = {
            metric: {"by_seed": dict(zip(SEEDS, scores, strict=True)),
                     "mean": mean(scores), "sample_sd": stdev(scores)}
            for metric, scores in values.items()
        }
    return result


def _labels(entry, test):
    return [{"label": name, "support": test["per_label_support"][index],
             "precision": test["per_label_precision"][index],
             "recall": test["per_label_recall"][index], "f1": test["per_label_f1"][index]}
            for index, name in enumerate(entry["label_order"])]


def _markdown(results, label_metrics, digest):
    rows, summary = results["runs"], results["three_seed_summary"]
    pair = results["modernbert_paired_comparison"]
    labels = {"macro_f1": "Macro-F1", "micro_f1": "Micro-F1",
              "subset_accuracy": "Subset Accuracy", "accuracy": "Accuracy",
              "positive_precision": "正类 Precision", "positive_recall": "正类 Recall",
              "positive_f1": "正类 F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
              "validation_macro_f1": "验证 Macro-F1"}
    lines = [
        "# Batch B · Encoder 正式测试报告", "",
        (f"正式冻结 SHA-256 `{digest}`。每个 Run 按自己的验证最优 Epoch、"
        "检查点与阈值完成测试。GoEmotions 使用 5,427 条测试样本，中文电影评论使用 "
        "369 条测试样本，其中语义正类 182 条、负类 187 条。"), "",
        ("[冻结清单](freeze.json) · [独立冻结审计](freeze-audit.json) · "
        "[验收](acceptance.json) · [结果明细](results.json) · [逐标签指标](label-metrics.json)"),
        "", "## 12 个冻结 Run", "",
        ("| 模型 | Seed | 来源 | 最佳 Epoch | 阈值 | 验证 Macro-F1 | 测试 Macro-F1 | "
        "测试 Micro-F1 | Subset Accuracy / Accuracy |"),
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for seed in SEEDS:
            row = rows[f"{model}-{seed}"]
            test = row["test"]
            accuracy = test["accuracy" if model == "B4" else "subset_accuracy"]
            lines.append(
                f'| {model} | {seed} | {row["stage"]} | {row["best_epoch"]} | '
                f'{row["threshold"]:.2f} | {row["validation_macro_f1"]:.4f} | '
                f'{test["macro_f1"]:.4f} | {test["micro_f1"]:.4f} | {accuracy:.4f} |'
            )
    lines += ["", "## 三种子稳定性", "",
              "| 模型 | 指标 | 42 | 123 | 2026 | 均值 ± 样本标准差 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for model, metrics in summary.items():
        for key, item in metrics.items():
            result = " | ".join(f'{item["by_seed"][seed]:.4f}' for seed in SEEDS)
            lines.append(f'| {model} | {labels[key]} | {result} | '
                         f'{item["mean"]:.4f} ± {item["sample_sd"]:.4f} |')
    lines += [
        "", "## ModernBERT-base 与 ModernBERT-large 配对比较", "",
        ("差值方向为 B3−B2。按样本 ID 对齐后对整行标签与预测成对重采样，"
        "10,000 次，报告 95% 百分位区间。三种子平均共享每次重采样的行索引；"
        "区间描述测试样本抽样波动。"), "",
        "| Seed | 指标 | B3−B2 | 95% 区间 |", "| --- | --- | ---: | --- |",
    ]
    for name, group in {**pair["by_seed"], "固定三种子平均": pair["mean_of_fixed_seeds"]}.items():
        for key, item in group.items():
            low, high = item["confidence_interval"]
            lines.append(f'| {name} | {labels[key]} | {item["difference"]:+.4f} | '
                         f'[{low:+.4f}, {high:+.4f}] |')
    lines += [
        "", "## B4 中文电影评论", "",
        ("真实检查点使用两维分类头和交叉熵。评分取第二维 Softmax 概率（索引 1）。"
        "原始标签 0 映射到语义正类 1，原始标签 1 映射到负类 0。"
        "混淆矩阵顺序为 TN、FP、FN、TP。"), "",
        "| Seed | TN | FP | FN | TP | 正类 P/R/F1 | 负类 F1 |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for seed in SEEDS:
        row = rows[f"B4-{seed}"]["test"]
        (tn, fp), (fn, tp) = row["confusion_matrix"]
        score = "/".join(f"{row[key]:.4f}" for key in CHINESE[2:5])
        negative = label_metrics[f"B4-{seed}"][0]["f1"]
        lines.append(f"| {seed} | {tn} | {fp} | {fn} | {tp} | {score} | {negative:.4f} |")
    lines += ["", "ROC-AUC 使用语义正类概率。PR-AUC 按项目指标契约使用 Average Precision。"]
    return "\n".join(lines) + "\n"


def publish(frozen, digest, datasets, progress):
    data, samples = test_inputs(frozen, datasets)
    rows, predictions, label_metrics = {}, {}, {}
    tested = OUTPUT / "test" / digest
    for index, entry in enumerate(frozen["runs"], start=1):
        key, name = run_key(entry), entry["dataset"]
        folder = tested / key
        metrics, pred = read_test(folder, entry, frozen, digest, data[name], samples[name])
        rows[key] = {field: entry[field] for field in (
            "model", "seed", "dataset", "stage", "role", "best_epoch", "threshold",
            "validation_macro_f1", "score_kind", "run",
        )} | {"test": metrics, "test_evidence": folder.as_posix(),
             "test_manifest_sha256": sha256_file(folder / "evaluation_manifest.json")}
        predictions[key] = pred
        label_metrics[key] = _labels(entry, metrics)
        progress.emit("REPLAY", f"{key} metrics and per-sample results verified",
                      completed=index, total_runs=len(frozen["runs"]))
    truth = data["goemotions"].encode(samples["goemotions"]).astype(bool)
    progress.emit("BOOTSTRAP", "Computing aligned B3-B2 test comparisons")
    results = {
        "freeze_sha256": digest, "runs": rows, "three_seed_summary": _summary(rows),
        "modernbert_paired_comparison": _paired(truth, predictions, rows,
                                                frozen["protocol"]["bootstrap"]),
    }
    folder = REPORTS / digest[:12]
    evidence = (OUTPUT / "freeze_audit.json").read_bytes()
    files = {"freeze.json": (OUTPUT / "frozen" / f"{digest}.json").read_bytes(),
             "freeze-audit.json": evidence, "results.json": json_bytes(results),
             "label-metrics.json": json_bytes(label_metrics),
             "report.md": _markdown(results, label_metrics, digest).encode("utf-8")}
    for name, content in files.items():
        publish_once(folder / name, content)
    audit = read_json(OUTPUT / "freeze_audit.json")
    acceptance = {
        "status": "PASS", "freeze_sha256": digest, "runs_verified": len(rows),
        "validation_rows_verified": sum(row["validation_rows"] for row in audit["runs"].values()),
        "test_rows_verified": sum(len(samples[row["dataset"]]) for row in rows.values()),
        "per_run_test_manifest_sha256": {key: row["test_manifest_sha256"]
                                         for key, row in rows.items()},
        "report_sha256": {name: sha256_file(folder / name) for name in files},
    }
    publish_once(folder / "acceptance.json", json_bytes(acceptance))
    return {"validated_test_runs": len(rows), "freeze_sha256": digest, "report": folder.as_posix()}
