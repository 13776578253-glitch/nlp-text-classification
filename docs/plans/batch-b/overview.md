# Batch B · 预训练 Transformer 文本分类

Batch B 对应实验矩阵 E4，沿用 Batch A 的两个 v1 数据集、标签语义、固定分区和验证 Macro-F1。S1 的跨批对照使用 Batch A 原冻结记录里的验证指标。正式测试另行冻结模型、阈值、评估代码与运行清单。

| 阶段 | 工作 | 完成依据 |
| --- | --- | --- |
| [S1 Encoders](s1-encoders.md) | B1–B4 完整微调，一个基础配置和一个学习率候选 | 资源核验、四组 CUDA 冒烟、八个训练槽位、独立验证审计和报告 |
| [S2 补充实验](s2-followups.md) | 根据 S1 的实测问题制定小规模矩阵 | 明确的现象、证据、待验证假设和预算 |
| [S3 Qwen 提案](s3-qwen-proposal.md) | 独立的 LLM 分类器研究 | 后续另定训练、预算和验收协议 |

当前执行范围止于 S1 的验证报告与 S2 评估入口。Batch A 正式产物归档于 [研究总结与模型索引](../../finish-task/batch-a/summary.md)。

[四组模型资源验收](model-resources.md)
