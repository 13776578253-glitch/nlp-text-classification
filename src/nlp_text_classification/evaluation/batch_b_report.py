"""Publish audited Batch B S1 validation results and evidence-linked follow-ups."""

from __future__ import annotations

import csv
from pathlib import Path

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.evaluation.batch_a_s1_report import _table
from nlp_text_classification.experiments.batch_b_s1 import FORMAL, _identity, _verify_attempt
from nlp_text_classification.experiments.suite import (
    checked_file,
    load_training_data,
    load_yaml,
    read_json,
)
from nlp_text_classification.training.progress import write_atomic_json

REPORT = Path("reports/batch-b-s1")


def _history(run):
    with (Path(run) / "history.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _plot(target, key, slots, selected):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(7.1, 4.8), sharex=True, layout="constrained")
    for case, color in (("baseline", "tab:blue"), ("alternate", "tab:orange")):
        slot = f"{case}/{key}"
        entry = slots[slot]
        history = _history(entry["run"])
        epochs = [int(row["epoch"]) for row in history]
        for axis, field in zip(axes, ("val_macro_f1", "loss"), strict=True):
            axis.plot(epochs, [float(row[field]) for row in history], marker="o", color=color,
                      label=f"{case} · LR {entry['settings']['learning_rate']:.0e}")
        if slot == selected:
            result = entry["result"]
            axes[0].scatter(result["best_epoch"], result["macro_f1"],
                            color=color, marker="*", s=160, zorder=4, label="Selected epoch")
    axes[0].set_title(f"Batch B S1 | {key} | seed 42")
    axes[0].set_ylabel("Validation Macro-F1")
    axes[1].set_ylabel("Training loss")
    axes[1].set_xlabel("Epoch")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="lower right", fontsize=8)
    figure.savefig(target, dpi=160)
    plt.close(figure)


def _comparators():
    path = Path("artifacts/batch-a/final_runs.yaml")
    digest = "255a9bfe53eea86f1339b2de4da46b96bfd9ab1594b723de5da11acf678c3289"
    base = load_yaml(checked_file(path, digest))
    return {key: entry["validation_macro_f1"] for key, entry in base["main"].items()}


def _diagnostics(suite, selected, slots):
    findings = []
    for key, slot in selected.items():
        row = slots[slot]
        result, settings = row["result"], row["settings"]
        run = row["run"]
        if result["stop_reason"] == "epoch_cap" and result["best_epoch"] >= settings["epochs"] - 1:
            findings.append({"model": key, "run": run, "issue": "convergence",
                             "evidence": f"最佳轮次 {result['best_epoch']}/{settings['epochs']}",
                             "followup": "固定已选配置做单模型延长，再按验证趋势决定复验。"})
        metrics = row["validation"]
        if metrics["modal_prediction_share"] >= 0.99 or metrics["predicted_label_fraction"] < 0.25:
            findings.append({"model": key, "run": run, "issue": "prediction_degeneracy",
                             "evidence": (f"众数预测占比 {metrics['modal_prediction_share']:.3f}，"
                                          f"标签覆盖 {metrics['predicted_label_fraction']:.3f}"),
                             "followup": "逐类核对验证支持量、原始分数、标签顺序和所选阈值。"})
        cases = [slots[f"{case}/{key}"]["settings"] for case in ("baseline", "alternate")]
        micro_batches = [case["micro_batch_size"] for case in cases]
        if len(set(micro_batches)) != 1:
            findings.append({"model": key, "run": run, "issue": "resource_adjustment",
                             "evidence": f"基础与候选微批次 {micro_batches}",
                             "followup": "用两种配置共用的可运行微批次复核关键结果。"})
    english = sorted(
        (key for key, model in suite["models"].items() if model["dataset"] == "goemotions"),
        key=lambda key: slots[selected[key]]["result"]["macro_f1"], reverse=True,
    )
    a, b = english[:2]
    gap = slots[selected[a]]["result"]["macro_f1"] - slots[selected[b]]["result"]["macro_f1"]
    if gap <= 0.01:
        findings.append({"model": f"{a} / {b}", "run": slots[selected[a]]["run"],
                         "issue": "seed_stability", "evidence": f"前两名验证差 {gap:.4f}",
                         "followup": "使用原入选超参数和固定追加种子复验排序稳定性。"})
    return findings


def publish(suite, datasets, tokenization, queue, progress=None):
    models = suite["models"]
    expected = {f"{case}/{key}" for case in ("baseline", "alternate") for key in models}
    if set(queue["accepted"]) != expected or queue["identity"] != _identity(suite, datasets):
        raise ValueError("S1 publication requires eight accepted slots under the frozen protocol")
    data = {key: load_training_data(item) for key, item in datasets.items()}
    slots = {}
    for slot, entry in queue["accepted"].items():
        run = Path(entry["run"])
        key = slot.split("/")[1]
        if progress:
            progress.context = {"slot": slot, "accepted_slots": len(queue["accepted"]),
                                "total_slots": len(expected), "audited_slots": len(slots)}
            progress.emit("AUDIT", f"{slot} replaying train and validation scores")
        if not run.resolve().is_relative_to((FORMAL / "runs").resolve()):
            raise ValueError(f"Unexpected result location: {run}")
        verified = _verify_attempt(run, data[models[key]["dataset"]])
        if verified != entry:
            raise ValueError(f"Final audit differs from the accepted slot: {slot}")
        manifest = read_json(run / "run_manifest.json")
        cfg = load_yaml(run / "resolved_config.yaml")
        planned_lr = (suite["training"]["baseline_lr"] if slot.startswith("baseline/") else
                      models[key]["alternate_lr"])
        if (manifest["cohort"], manifest["seed"], cfg["settings"]["learning_rate"]) != (
            slot, suite["seed"], planned_lr
        ) or cfg["inputs"] != queue["identity"]:
            raise ValueError(f"Run was trained under another protocol: {slot}")
        slots[slot] = {**entry, "settings": cfg["settings"],
                       "validation": read_json(run / "metrics_validation.json"),
                       "manifest": manifest}
    selected = {}
    for key in models:
        base, alternate = (f"{case}/{key}" for case in ("baseline", "alternate"))
        delta = slots[alternate]["result"]["macro_f1"] - slots[base]["result"]["macro_f1"]
        selected[key] = alternate if delta > suite["selection"]["score_tie_atol"] else base
    comparisons = _comparators()
    findings = _diagnostics(suite, selected, slots)
    write_atomic_json(FORMAL / "audit.json", {
        "status": "PASS", "splits_used": ["train", "validation"],
        "queue_sha256": sha256_file(FORMAL / "queue.json"),
        "selected": selected, "runs": {slot: entry["files"] for slot, entry in slots.items()},
    })
    REPORT.mkdir(parents=True, exist_ok=True)
    target = REPORT / "curves"
    target.mkdir(exist_ok=True)
    rows = []
    for key, slot in selected.items():
        _plot(target / f"{key}.png", key, slots, slot)
        model = models[key]
        entry = slots[slot]
        result = entry["result"]
        baseline = comparisons[f"{model['dataset']}/E1"]
        inference = result["validation_inference"]
        rows.append([f"[{key}](curves/{key}.png)", model["model_id"], model["dataset"],
                     f'{result["macro_f1"]:.4f}', f"{baseline:.4f}",
                     f'{result["macro_f1"] - baseline:+.4f}',
                     f'{result["best_epoch"]}/{result["epochs_completed"]}',
                     f'{result["parameters"]:,}',
                     f'{inference["samples"] / inference["seconds"]:.0f}'])
    lines = [
        "# EmotionLab · Batch B S1 验证报告", "",
        ("四个官方 Encoder 以 seed 42 完整微调，每个模型比较一个基础配置和一个学习率候选。"
         "本报告使用冻结 v1 的训练与验证分区。Batch A 参照来自原冻结清单的验证分数。"), "",
        "## 入选结果", "",
        *_table(["配置与曲线", "模型", "数据集", "验证 Macro-F1", "Batch A E1 验证 F1",
                 "差值", "最佳/实际 Epoch", "参数量", "推理样本/秒"], rows), "",
        "## 全部训练槽位", "",
    ]
    rows = []
    for slot, entry in slots.items():
        result, settings = entry["result"], entry["settings"]
        rows.append([slot, Path(entry["run"]).name, f'{settings["learning_rate"]:.0e}',
                     f'{result["macro_f1"]:.4f}', f'{result["threshold"]:.2f}',
                     f'{result["best_epoch"]}/{result["epochs_completed"]}',
                     result["stop_reason"], str(settings["micro_batch_size"]),
                     str(settings["effective_batch_size"]), f'{result["elapsed_seconds"] / 60:.2f}',
                     f'{result["peak_allocated_bytes"] / 1024**3:.2f}',
                     f'{result["peak_reserved_bytes"] / 1024**3:.2f}',
                     str(result["validation_inference"]["batch_size"])])
    lines.extend(_table(["槽位", "Run ID", "LR", "验证 F1", "阈值", "最佳/实际 Epoch",
                         "停止原因", "微批次", "有效批次", "分钟", "峰值 allocated GiB",
                         "峰值 reserved GiB", "推理批次"], rows))
    lines.extend(["", "## Tokenization 与证据", ""])
    for key, item in tokenization.items():
        rates = [item["partitions"][split]["truncated_fraction"] for split in ("train", "validation")]
        lines.append(f'- {key} · {item["selected_length"]} token，'
                     f'训练/验证截断比例 {rates[0]:.2%} / {rates[1]:.2%}。')
    lines.extend(["", ("完整指标和数据、模型、Tokenizer 指纹见 `results.json`。"
                       "训练状态、逐样本预测和审计见 `artifacts/batch-b-s1/formal/`。"),
                  "", "## S2 评估入口", ""])
    for item in findings:
        lines.append(f'- {item["model"]} · {item["evidence"]}。{item["followup"]}')
    if not findings:
        lines.append("按本次筛查条件，暂无需立即补跑的模型。")
    lines.extend(["", "S2 实际矩阵待评估。正式测试与 S3 为独立阶段。", ""])
    (REPORT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_atomic_json(REPORT / "results.json", {
        "identity": queue["identity"], "hardware": queue["hardware"], "selected": selected,
        "validation_only": True, "runs": slots, "tokenization": tokenization,
        "batch_a_validation": comparisons, "s2_followups": findings,
        "attempts": queue["attempts"],
    })
    return {"report": str(REPORT / "report.md"), "selected": selected, "s2_findings": len(findings)}
