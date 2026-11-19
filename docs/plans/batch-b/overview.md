# Batch B · 预训练 Transformer 文本分类

Batch B 对应实验矩阵 E4，沿用 Batch A 的两个 v1 数据集、标签语义、固定分区和验证 Macro-F1。S1 的跨批对照使用 Batch A 原冻结记录里的验证指标。正式测试另行冻结模型、阈值、评估代码与运行清单。

| 阶段 | 工作 | 完成依据 |
| --- | --- | --- |
| [S1 Encoders](s1-encoders.md) | B1–B4 完整微调，一个基础配置和一个学习率候选 | 资源核验、四组 CUDA 冒烟、八个训练槽位、独立验证审计和报告 |
| [S2 补充实验](s2-followups.md) | 根据 S1 的实测问题制定小规模矩阵 | 明确的现象、证据、待验证假设和预算 |
| [Encoder 正式冻结与测试](final-test.md) | 独立审计并测试 B1–B4 的 12 个入选 Run | 内容寻址清单、逐样本测试证据、指标复算与研究报告 |
| [S3 Qwen 预研及预检](s3-qwen-proposal.md) | Qwen3.5-4B-Base 的 NF4 QLoRA 多标签分类与 WSL2 CUDA 加速 | 官方模型与 Tokenizer 核验、覆盖审计、有限更新与恢复 |
| [S4 Qwen 正式实验](../../finish-task/batch-b-s4.md) | 优化正式入口，三种子训练、验证冻结与独立测试 | 三次已冻结测试、逐样本复算、正式报告与证据索引 |

Encoder 正式测试是 S1、S2 的交付阶段。S3 的工程证据由 S4 接续，S4 采用独立的 Qwen 冻结清单。Batch A 正式产物归档于 [研究总结与模型索引](../../finish-task/batch-a/summary.md)。

[四组模型资源验收](model-resources.md)
