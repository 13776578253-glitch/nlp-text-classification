"""Report the S2 candidate screen and verified checkpoint continuations."""

from __future__ import annotations

from pathlib import Path

from nlp_text_classification.evaluation.batch_a_s1_report import _table
from nlp_text_classification.experiments.batch_a_s1 import read_history
from nlp_text_classification.experiments.batch_a_s2 import OUTPUT
from nlp_text_classification.training.progress import write_atomic_json

DESTINATION = Path("reports/batch-a-s2")
SEED_COLORS = {42: "tab:blue", 123: "tab:orange", 2026: "tab:green"}


def _plot(path, model, rows, cap, *, screening):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.1, 4.9), layout="constrained")
    for seed, row in rows.items():
        color = SEED_COLORS[seed]
        data = row if screening else row["s2"]
        history = read_history(data["run"])
        points = history[12:] if screening else history
        for axis, field in zip(axes, ("val_macro_f1", "loss"), strict=True):
            epoch = [int(point["epoch"]) for point in points]
            values = [float(point[field]) for point in points]
            if screening:
                axis.plot(epoch, values, color=color, marker=".", label=f"seed {seed}")
            else:
                start = row["source_epoch"] - 1
                axis.plot(epoch[:start + 1], values[:start + 1], color=color, linestyle="--")
                axis.plot(epoch[start:], values[start:], color=color, label=f"seed {seed}")
        axes[0].scatter(data["best_epoch"], data["macro_f1"], color=color, s=23, zorder=4)
    for axis in axes:
        if screening:
            axis.axvspan(cap - 3, cap, color="0.92", zorder=0)
            axis.set_xlim(12.6, cap + 0.4)
        else:
            axis.axvline(cap, color="0.4", linestyle=":", linewidth=1)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Validation Macro-F1")
    axes[1].set_ylabel("Training BCE loss")
    axes[1].set_xlabel("Epoch")
    axes[0].set_title(f"GoEmotions  |  {model}  |  {'S1 tail' if screening else 'S2 continuation'}")
    axes[0].legend(ncol=len(rows), loc="lower right", fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _screening(experiment, screen, destination):
    target = destination / "s1-tails"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in experiment.s1.resolved:
        source = {seed: screen["records"][f"{model}/{seed}"] for seed in experiment.seeds}
        _plot(target / f"{model}.png", model, source, experiment.s1_cap, screening=True)
        for seed, item in source.items():
            score_tail = " → ".join(f'{row["val_macro_f1"]:.4f}' for row in item["tail"][-4:])
            rows.append([f"[{model}](s1-tails/{model}.png)", str(seed),
                         f'{item["best_epoch"]}/{item["epochs"]}', f'{item["macro_f1"]:.4f}',
                         score_tail, "入选依据" if item["rising_at_cap"] else "—"])
    lines = [
        "# Batch A Supplement S2 · GoEmotions 收敛边界", "",
        ("S1 给六种模型相同训练预算，以比较排名与消融。"
         "S2 在同一数据、词表、参数和验证口径下续训入选模型，"
         "用于定位各模型的验证平台期。两轮都属于 Batch A。"), "",
        "## S1 轨迹与候选筛选", "",
        ("当至少一个种子的最佳点位于第 21–24 轮，且最后四轮的验证 Macro-F1 "
         "线性斜率为正、均值高于第 17–20 轮时，模型入选。"
         "筛选覆盖六组配置的全部 18 条 S1 轨迹。"), "",
        *_table(["模型与末段曲线", "Seed", "最佳/实际轮次", "S1 最佳 F1", "末四轮 F1", "筛选"], rows),
        "",
    ]
    lines.extend(f'- {model} · 入选种子 {", ".join(map(str, seeds))}'
                 for model, seeds in screen["candidates"].items())
    if not screen["candidates"]:
        lines.append("筛选完成，候选数为 0。")
    return lines


def _status(item):
    if item["stop_reason"] == "early_stopping":
        return "平台期，连续 6 轮未刷新最佳"
    if item["rising_at_cap"]:
        return "抵达 36 轮，验证仍上升"
    return "抵达 36 轮，尾段未达到上升条件"


def publish(experiment, screen, data):
    destination = DESTINATION if data is not None else OUTPUT
    lines = _screening(experiment, screen, destination)
    if data is not None:
        target = DESTINATION / "curves"
        target.mkdir(exist_ok=True)
        rows = []
        for model in screen["candidates"]:
            source = {seed: data["runs"][f"{model}/{seed}"] for seed in experiment.seeds
                      if f"{model}/{seed}" in data["runs"]}
            _plot(target / f"{model}.png", model, source, experiment.s1_cap, screening=False)
            for seed, info in source.items():
                old, new = info["s1"], info["s2"]
                rows.append([f"[{model}](curves/{model}.png)", str(seed), f'{old["macro_f1"]:.4f}',
                             f'{new["macro_f1"]:.4f}', f'{info["gain"]:+.4f}',
                             f'{new["best_epoch"]}/{new["epochs"]}', _status(new)])
        replicated = "、".join(data["replicate"]) or "无"
        more_budget = "、".join(data["recommend_more_budget"]) or "无"
        lines.extend([
            "", "## 最长 36 轮的续训结果", "",
            ("先运行 seed 42，再对验证 Macro-F1 增益达到 0.01 的模型补跑 seed 123、2026。"
             "已执行的追加种子沿用各自 S1 检查点与同样的 patience 6。"), "",
            *_table(["模型与续训曲线", "Seed", "S1 最佳 F1", "S2 最佳 F1", "增益",
                     "S2 最佳/实际轮次", "收敛状态"], rows),
            "", f"- 达到 0.01 门槛并复验的模型 · {replicated}",
            f"- 有种子在第 36 轮仍达到上升条件，若要定位平台期需另行增加预算 · {more_budget}",
            "", "## 恢复与审计", "",
            ("S2 恢复最新轮次的模型权重、Adam 优化器、采样器及 CPU/CUDA RNG，"
             "并继承原最佳检查点和早停计数。原训练使用固定学习率。"
             "继承的逐轮记录与所用状态源逐项一致。"),
            ("S2 使用独立的运行目录、队列和文件指纹。"
             "每个 Run 的恢复来源、轮次、完整检查点 SHA-256 和末段数据见 `results.json`。"),
        ])
        write_atomic_json(DESTINATION / "results.json", data)
    path = destination / "report.md"
    path.write_text("\n".join([*lines, ""]), encoding="utf-8")
    return {"report": str(path), "candidates": screen["candidates"],
            "replicate": data["replicate"] if data else [],
            "recommend_more_budget": data["recommend_more_budget"] if data else []}
