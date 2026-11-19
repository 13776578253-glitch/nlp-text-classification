# Batch B S4 完成归档

S4 接续 Qwen3.5-4B-Base 的 S3 技术预研。采用同一官方 Revision、NF4 QLoRA、248 个投影和 28 维分类头，在 GoEmotions 固定分区完成三次独立训练。优化路径、训练与正式测试均有可复核的记录。 [S4 性能调查](../../reports/batch-b-s4/performance/report.md) · [工程入口验收](../../reports/batch-b-s4/formal/report.md)

| 阶段 | 交付与证据 |
| --- | --- |
| 正式训练 | Seed 42、123、2026，共 11 个完整 Epoch，分别在 4、4、3 轮早停。 [验证结果](../../reports/batch-b-s4/formal/report-s4.md) |
| 验证冻结 | 三组最佳检查点与阈值冻结为 `97ce2926…`，冻结阶段 `test_read: false`。 [冻结引用](../../reports/batch-b-s4/formal/freeze-ref.json) · [冻结审计](../../reports/batch-b-s4/formal/freeze-audit.json) |
| 独立正式测试 | 3/3 Run 复算通过。一次读取源测试分区，5,427 条样本按冻结顺序对齐，每个 Seed 使用自己的冻结阈值。 [正式报告](../../reports/batch-b-s4/formal/final-test.md) · [测试签收](../../reports/batch-b-s4/formal/test-acceptance.json) |
| 复现和报告 | [Runbook](../runbooks/batch-b-s4-formal.md) · [训练曲线](../../reports/batch-b-s4/formal/training-curves.png) · [LaTeX 结果表](../../reports/batch-b-s4/formal/test-results.tex) |

正式测试使用与 Batch B 英文 Encoder 一致的 GoEmotions 测试分区。三种子的标准差描述该配置的训练波动。固定阈值下，测试 Micro-F1、Subset Accuracy 与各类支持度一并保留，供后续解读 Macro-F1。后续阶段的研究问题和范围待讨论。
