# 当前任务状态 · Batch B S1

Batch B S1 计划与矩阵已冻结。当前入口为官方模型资源准备。见 [Batch B 总览](../plans/batch-b/overview.md) 与 [S1 契约](../plans/batch-b/s1-encoders.md)。

Batch A Stage 0–5、Supplement S1 和 S2 均已完成。见 [统一研究总结](../finish-task/batch-a/summary.md) 与 [模型结果索引](../finish-task/batch-a/model-index.json)。

| 项目 | 状态与证据 |
| --- | --- |
| 前置迁移 | 已完成。提交 `ac0253e` 的父提交为历史整理后的 `543a58f`。本地迁移测试 8 项、Ruff 通过。 |
| 执行方案修订 | 已完成文档和配置修订。6 份 YAML、14 个主结果位、预算算式、12 份 Markdown 的 40 处本地链接和锚点核对通过。预算、质量验收、复验和 Stage 0–5 依赖见执行计划。 |
| Stage 0 | 双数据 v1 实物审计、派生冻结与 9 项测试通过。见 [Stage 0 验收](../stage0-data-audit.md)。 |
| Stage 1 | 训练、Optuna、冻结审计与报告链路已用独立 CPU 合成批次验证。两个真实 Word2Vec 已读取或生成。 |
| Stage 2 | [CPU 验收](../stage2-smoke.md) 与 CUDA 12.8 全矩阵 14/14 冒烟均通过。 |
| Stage 3 | [正式训练验收](../stage3-training.md) 已通过。主 Run 14 个、必选复验 30 个、追加复验 3 个。 |
| Stage 4 | [独立审计与正式测试](../stage4-final-evaluation.md) 已通过。47 个 Run 冻结后全部评估。 |
| Stage 5 | [报告交付](../stage5-delivery.md) 已验收。18 份输出重建逐字节一致。 |
| Supplement S1 | [GoEmotions 延长训练](../runbooks/batch-a-s1.md) 已验收。6 个主对照和 12 个多种子复验通过，前 12 轮完整状态 6/6 精确重现。 |
| Supplement S2 | [GoEmotions 收敛边界](../runbooks/batch-a-s2.md) 已验收。筛选 18 条 S1 轨迹，4 个续训通过。Vanilla RNN 在第 36 轮仍有改善，LSTM 在第 30 轮早停。 |

查看批次进度的命令与证据路径见 [训练进度](../runbooks/batch-a-progress.md)。正式训练与审计状态以对应批次的进度文件及原始日志为准。

[执行计划](../plans/2026-11-17-batch-a-execution.md) · [证据契约](../contracts/ARTIFACTS.md) · [下一步](../NEXT.md)
