# 当前任务状态 · Batch A V2

| 项目 | 状态与证据 |
| --- | --- |
| 前置迁移 | 已完成。提交 `ac0253e` 的父提交为历史整理后的 `543a58f`。本地迁移测试 8 项、Ruff 通过。 |
| 执行方案修订 | 已完成文档和配置修订。6 份 YAML、14 个主结果位、预算算式、12 份 Markdown 的 40 处本地链接和锚点核对通过。预算、质量验收、复验和 Stage 0–5 依赖见执行计划。 |
| Stage 0 | 双数据 v1 实物审计、派生冻结与 9 项测试通过。见 [Stage 0 验收](../stage0-data-audit.md)。 |
| Stage 1 | 训练、Optuna、冻结审计与报告链路已用独立 CPU 合成批次验证。两个真实 Word2Vec 已读取或生成。 |
| Stage 2 | [CPU 验收](../stage2-smoke.md) 已通过。CUDA 预检与 GPU 冒烟待运行时安装完成。 |
| Stage 3 | 待执行。依赖全套 14 项冒烟验收。常规 GPU 84–88 次，含 HPO 补充和性能诊断最多 120 次。 |
| Stage 4 / 5 | 待执行。Stage 4 依赖主实验和复验质量、独立审计与完整冻结清单，Stage 5 依赖正式测试产物。 |

查看批次进度的命令与证据路径见 [训练进度](../runbooks/batch-a-progress.md)。正式训练与审计状态以对应批次的进度文件及原始日志为准。

[执行计划](../plans/2026-11-17-batch-a-execution.md) · [证据契约](../contracts/ARTIFACTS.md) · [下一步](../NEXT.md)
