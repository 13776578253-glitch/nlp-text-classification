# Batch B S2 · 验收报告

2026-11-18，RTX 4070 Ti SUPER 串行完成九个新增 Run。B1 预算对照和八个种子复验均通过。四个模型的三种子比较共 12 个 Run，沿用各自通过验证的检查点和阈值。

| 验收项 | 实际结果 |
| --- | --- |
| 输入及原产物 | 复核八个 S1 Run、模型和 Tokenizer 锁、train/validation 指纹。Batch A 原冻结结果、47 份正式测试和原报告通过指纹核对。 |
| 配置与预算 | B1 的 12 轮独立协议在第 7 轮早停，最佳验证 Macro-F1 为 0.4918，较原 6 轮协议低 0.0040，保留原协议。四个模型的追加 Run 逐项核对了种子、调度步数、配置和获选阈值。 |
| 预测及指标 | 新增九个 Run 均通过模型重载和独立复算。逐样本分数复原为 float32，重算验证 Macro-F1、全局阈值网格和早停。 |
| 三种子与标签 | 每个配置均有 42、123、2026 三项验证结果。核对样本标准差、逐标签支持量和低支持组的差值分解。Seed 123 的 B1、B2 Subset Accuracy 以并列显示。 |
| 拟冻结结果 | 4 个 seed 42 主 Run 与 8 个种子复验 Run。统一绑定权重、配置、Tokenizer、标签和阈值，状态为 `proposed`。 |
| 重建与代码 | 五个报告文件重建逐字节一致。34 项 pytest、Ruff 和文档链接检查通过。正式训练引用提交 `28083fa`。 |

独立验收证据在 `artifacts/batch-b-s2/input-audit.json`、`delivery-audit.json` 和 `acceptance-check.json`。训练状态、日志、逐样本分数、恢复与最优检查点在各 Run 目录。正式测试待独立冻结审计，S3 保持独立规划。

[S2 研究报告](../../reports/batch-b-s2/report.md) · [逐标签结果](../../reports/batch-b-s2/label-metrics.json) · [拟冻结清单](../../reports/batch-b-s2/proposed-freeze.json) · [运行手册](../archive/runbooks/batch-b-s2.md)
