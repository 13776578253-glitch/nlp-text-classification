"""Compare the frozen Batch A validation results with the S1 learning curves."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

from nlp_text_classification.experiments.batch_a_s1 import (
    ABLATION,
    CURVE_FIELDS,
    OUTPUT,
    read_history,
)
from nlp_text_classification.experiments.suite import load_yaml, read_json
from nlp_text_classification.training.progress import write_atomic_json


def compare_runs(experiments, originals, extended, anchor, trigger, tolerance):
    order = ("E1", *experiments)
    before, after = {"E1": anchor}, {"E1": anchor}
    for experiment in experiments:
        before[experiment] = originals[experiment]["macro_f1"]
        after[experiment] = extended[experiment]["macro_f1"]
    # Stable sorting retains the matrix order for a tied score.
    ranked = [sorted(order, key=scores.__getitem__, reverse=True) for scores in (before, after)]
    flips = [
        [a, b] for a, b in combinations(order, 2)
        if (before[a] - before[b]) * (after[a] - after[b]) < 0
        and min(abs(before[a] - before[b]), abs(after[a] - after[b])) > tolerance
    ]
    reasons = {}
    for experiment in experiments:
        current = []
        if after[experiment] - before[experiment] + tolerance >= trigger["minimum_gain"]:
            current.append("验证 Macro-F1 提升至少 0.02")
        if extended[experiment]["best_epoch"] >= trigger["first_near_cap_epoch"]:
            current.append("最佳轮次落在第 22–24 轮")
        for a, b in flips:
            if experiment in (a, b):
                current.append(f"{a} 与 {b} 的排名互换")
        if current:
            reasons[experiment] = current
    if set(ABLATION) & reasons.keys():
        for experiment in ABLATION:
            reasons.setdefault(experiment, ["随 TextCNN 三组受控消融复验"])
    return {"rank_before": ranked[0], "rank_after": ranked[1], "rank_flips": flips,
            "reasons": reasons, "replicate": [name for name in experiments if name in reasons]}


def reference_result(history, cap, patience, tolerance):
    best_score, best_epoch = -1.0, 0
    for row in history[:cap]:
        score, epoch = float(row["val_macro_f1"]), int(row["epoch"])
        if score > best_score + tolerance:
            best_score, best_epoch = score, epoch
        if epoch - best_epoch >= patience:
            break
    return {"macro_f1": best_score, "best_epoch": best_epoch, "epochs": epoch}


def _summary(run):
    result = read_json(Path(run) / "result.json")
    history = read_history(run)
    return {name: result[name] for name in ("macro_f1", "best_epoch", "stop_reason")} | {
        "run": str(run), "epochs": len(history),
        "last_three_f1": [float(row["val_macro_f1"]) for row in history[-3:]],
        "last_three_loss": [float(row["loss"]) for row in history[-3:]],
    }


def _replications(supplement, runs):
    rows = {}
    cap = supplement.spec["reference_epoch_cap"]
    tolerance = supplement.config["selection"]["score_tie_atol"]
    for key, run in runs.items():
        experiment = key.split("/")[0]
        old_config = supplement.originals[experiment] / "resolved_config.yaml"
        patience = load_yaml(old_config)["settings"]["patience"]
        history = read_history(run)
        baseline = reference_result(history, cap, patience, tolerance)
        frozen = supplement.frozen["replications"].get(f"{supplement.dataset}/{key}")
        if frozen:
            original = read_history(supplement._verify_entry(frozen))
            if len(original) != baseline["epochs"] or any(
                any(a[field] != b[field] for field in CURVE_FIELDS)
                for a, b in zip(original, history[:baseline["epochs"]], strict=True)
            ):
                raise ValueError(f"Frozen seed trajectory differs for {key}")
        extended = _summary(run)
        rows[key] = {"reference": baseline, "extended": extended,
                     "frozen_prefix_exact": True if frozen else None,
                     "gain": extended["macro_f1"] - baseline["macro_f1"]}
    return rows


def _conclusion(supplement, primary, replications):
    trigger = supplement.spec["replication_trigger"]
    tolerance = supplement.config["selection"]["score_tie_atol"]
    seeds = supplement.spec["replication_seeds"]
    material, near_cap = [], []
    for model in primary["replicate"]:
        before, after = primary["before"][model], primary["after"][model]
        rows = [replications[f"{model}/{seed}"] for seed in seeds]
        gains = [after["macro_f1"] - before["macro_f1"], *(row["gain"] for row in rows)]
        epochs = [after["best_epoch"], *(row["extended"]["best_epoch"] for row in rows)]
        if sum(gain + tolerance >= trigger["minimum_gain"] for gain in gains) >= 2:
            material.append(model)
        if sum(epoch >= trigger["first_near_cap_epoch"] for epoch in epochs) >= 2:
            near_cap.append(model)
    repeated_flips = []
    for a, b in primary["rank_flips"]:
        direction = primary["after"].get(a, {"macro_f1": supplement.anchor})["macro_f1"] - (
            primary["after"].get(b, {"macro_f1": supplement.anchor})["macro_f1"]
        )
        repeated = []
        for seed in seeds:
            scores = []
            for stage in ("reference", "extended"):
                scores.append([
                    supplement.anchor if model == "E1" else
                    replications[f"{model}/{seed}"][stage]["macro_f1"]
                    for model in (a, b)
                ])
            before, after = (left - right for left, right in scores)
            if before * after < 0 < after * direction and min(abs(before), abs(after)) > tolerance:
                repeated.append(seed)
        if repeated:
            repeated_flips.append({"models": [a, b], "additional_seeds": repeated})
    return {"recommend_new_version": bool(material or near_cap or repeated_flips),
            "material_gain_at_least_two_seeds": material, "near_cap_at_least_two_seeds": near_cap,
            "repeated_rank_flips": repeated_flips}


def _curve(original, extended, output, model, best):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.1, 4.9), layout="constrained")
    for axis, field, label in zip(
        axes, ("val_macro_f1", "loss"), ("Validation Macro-F1", "Training BCE loss"), strict=True
    ):
        for run, style, name in ((extended, "-", "S1 · up to 24 epochs"),
                                 (original, "--", "Batch A · 12 epochs")):
            points = read_history(run)
            axis.plot([int(row["epoch"]) for row in points],
                      [float(row[field]) for row in points], style, label=name)
        axis.axvline(12, color="0.45", linewidth=1, linestyle=":", label="Original budget")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    axes[0].scatter(best["best_epoch"], best["macro_f1"], zorder=4, color="tab:blue",
                    label="S1 best epoch")
    axes[0].set_title(f"GoEmotions  |  {model}  |  seed 42")
    axes[0].legend(ncol=2, fontsize=8, loc="lower right")
    axes[1].set_xlabel("Epoch")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _table(headers, rows):
    return ["| " + " | ".join(row) + " |" for row in (
        headers, ["---"] * len(headers), *rows,
    )]


def _report(supplement, primary, replications, conclusion):
    destination = Path("reports/batch-a-s1")
    curves = destination / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, after in primary["after"].items():
        before = primary["before"][model]
        _curve(before["run"], after["run"], curves / f"{model}.png", model, after)
        rows.append([f"[{model}](curves/{model}.png)", f'{before["macro_f1"]:.4f}',
                     f'{after["macro_f1"]:.4f}', f'{after["macro_f1"] - before["macro_f1"]:+.4f}',
                     f'{before["best_epoch"]}/{before["epochs"]}',
                     f'{after["best_epoch"]}/{after["epochs"]}', after["stop_reason"],
                     " → ".join(f"{score:.4f}" for score in after["last_three_f1"])])
    lines = [
        "# Batch A Supplement S1 · GoEmotions 收敛验证", "",
        ("S1 比较原 12 轮预算与最长 24 轮、patience 6 下的验证表现。"
        "六种神经配置沿用原入选参数、词表、训练与验证分区及全局阈值规则。"
        "E1 使用原验证成绩作为固定排名参照。"),
        "", "## Seed 42 对照", "",
        *_table(["配置与曲线", "原 Macro-F1", "S1 Macro-F1", "差值", "原最佳/实际轮次",
                 "S1 最佳/实际轮次", "停止原因", "末三轮验证 F1"], rows),
        "", "- 原排名 · " + " > ".join(primary["rank_before"]),
        "- S1 排名 · " + " > ".join(primary["rank_after"]),
        "- 排名互换 · " + ("、".join(" / ".join(pair) for pair in primary["rank_flips"]) or "无"),
        "", ("前 12 轮全部逐项精确重现。复算项包括训练 loss、梯度范数、验证 Macro-F1、"
        "阈值和最佳轮次。第 12 轮模型、Adam 状态、采样器和 CPU/CUDA 随机状态、"
        "最佳检查点与保存的验证分数也逐项一致。S1 从各自重现的检查点接续第 13 轮。"),
        "", "## TextCNN 词嵌入消融", "",
        ("三组共用非嵌入初始化、训练顺序及超参数。两种 Word2Vec 条件共享初始嵌入矩阵，"
        "冻结组通过逐字节检查。"),
        "",
    ]
    rows = []
    for a, b in (("E5-TUNE", "E2"), ("E5-FROZEN", "E2"), ("E5-TUNE", "E5-FROZEN")):
        before, after = (primary[stage][a]["macro_f1"] - primary[stage][b]["macro_f1"]
                         for stage in ("before", "after"))
        rows.append([f"{a} − {b}", f"{before:+.4f}", f"{after:+.4f}", f"{after-before:+.4f}"])
    lines.extend(_table(["对照", "原验证差值", "S1 验证差值", "差值变化"], rows))
    lines.extend(["", "## 复验触发", ""])
    for model, reasons in primary["reasons"].items():
        lines.append(f"- {model} · {'、'.join(reasons)}")
    if replications is None:
        lines.extend(["", "Seed 123 和 2026 待执行。"])
    else:
        lines.extend(["", "## 多种子验证", "",
                      ("追加种子在各自轨迹内，用原 12 轮上限与 patience 3 重放原选模口径。"
                      "有原冻结复验的配置逐轮核对共同前缀。"), ""])
        rows = []
        for model in primary["replicate"]:
            samples = [replications[f"{model}/{seed}"] for seed in supplement.spec["replication_seeds"]]
            values = [primary["after"][model]["macro_f1"],
                      *(row["extended"]["macro_f1"] for row in samples)]
            for seed, item in zip(supplement.spec["replication_seeds"], samples, strict=True):
                old, new = item["reference"], item["extended"]
                rows.append([model, str(seed), f'{old["macro_f1"]:.4f}', f'{new["macro_f1"]:.4f}',
                             f'{item["gain"]:+.4f}', f'{new["best_epoch"]}/{new["epochs"]}',
                             "精确一致" if item["frozen_prefix_exact"] else "—",
                             f"{mean(values):.4f} ± {stdev(values):.4f}"])
        lines.extend(_table(["配置", "种子", "原协议 F1", "S1 F1", "差值", "S1 最佳/实际轮次",
                             "原复验前缀", "S1 三种子均值 ± SD"], rows))
        lines.extend(["", "## 实验版本结论", ""])
        material = "、".join(conclusion["material_gain_at_least_two_seeds"]) or "无"
        cap = "、".join(conclusion["near_cap_at_least_two_seeds"]) or "无"
        flips = "、".join(f'{a} / {b}（seed {", ".join(map(str, flip["additional_seeds"]))}）'
                          for flip in conclusion["repeated_rank_flips"]
                          for a, b in [flip["models"]]) or "无"
        recommendation = ("建议为英文神经模型建立独立的长预算协议版本。"
                          if conclusion["recommend_new_version"] else "现有证据支持保留原协议版本。")
        lines.extend([recommendation, "", f"- 至少两个种子的提升达到 0.02 · {material}",
                      f"- 至少两个种子的最佳轮次仍在第 22–24 轮 · {cap}",
                      f"- seed 42 的排名互换在额外种子复现 · {flips}"])
    lines.extend(["", "## 审计与复现", "",
                  f'原冻结 SHA-256 `{supplement.inputs["freeze_sha256"]}`。',
                  ("S1 的运行、配置、日志、验证预测与检查点位于 `artifacts/batch-a-s1/`。"
                  "`queue.json` 绑定每个通过审计的运行与文件指纹，"
                  "`prefix_audit.json` 保存 12 轮前缀复核，`primary.json` 保存复验决策。"),
                  "`results.json` 汇总正式的 S1 验证分析。命令见项目实验索引。", ""])
    path = destination / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def publish(supplement, mode):
    if mode == "primary":
        runs = supplement.primary()
        before = {model: _summary(path) for model, path in supplement.originals.items()}
        after = {model: _summary(path) for model, path in runs.items()}
        decision = compare_runs(tuple(runs), before, after, supplement.anchor,
                                supplement.spec["replication_trigger"],
                                supplement.config["selection"]["score_tie_atol"])
        primary = {"inputs": supplement.inputs, "runs": runs, "before": before, "after": after,
                   **decision}
        path = OUTPUT / "primary.json"
        if path.exists() and read_json(path) != primary:
            raise ValueError("The published S1 primary decision changed")
        write_atomic_json(path, primary)
        supplement.progress.context = {}
        supplement.progress.emit("S1_PRIMARY_COMPLETE", "seed 42 audited",
                                 replication_triggers=decision["reasons"])
        if not (OUTPUT / "replications.json").exists():
            return {"report": str(_report(supplement, primary, None, None)), **decision}
    primary = supplement.read_primary()
    path = OUTPUT / "replications.json"
    if mode == "replicate":
        runs = supplement.replicate(primary)
        write_atomic_json(path, {"primary": primary, "runs": runs})
    if not path.exists():
        return {"report": str(_report(supplement, primary, None, None))}
    saved = read_json(path)
    if saved["primary"] != primary:
        raise ValueError("Replications refer to another S1 primary decision")
    for key, run in saved["runs"].items():
        if supplement._verify_entry(supplement.queue["audited"][f"replication/{key}"]) != Path(run):
            raise ValueError(f"Replication does not match S1 audit: {run}")
    records = _replications(supplement, saved["runs"])
    conclusion = _conclusion(supplement, primary, records)
    report = _report(supplement, primary, records, conclusion)
    results = {**primary, "replications": records, "conclusion": conclusion}
    write_atomic_json(report.parent / "results.json", results)
    supplement.progress.context = {}
    supplement.progress.emit("S1_COMPLETE", f'{len(primary["runs"])} primary, {len(records)} replication',
                             report=report)
    return {"report": str(report), **conclusion}
