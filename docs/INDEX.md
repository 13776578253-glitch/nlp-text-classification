# 项目文档

## 核心规范与交付索引

| 文档 | 职责 |
| --- | --- |
| [实验数据盘点与交付索引](experiment-data-inventory.md) | 双数据集全部分布、62 个正式测试 Run 证据路径、前端与报告取数入口 |
| [架构](ARCHITECTURE.md) | 目录、模块职责与执行依赖 |
| [数据契约](contracts/DATASETS.md) | 来源、分区、标签和预处理边界 |
| [实验矩阵](contracts/EXPERIMENT_MATRIX.md) | Batch A/B 范围、主结果位、消融复用和复验矩阵 |
| [实验契约](contracts/EXPERIMENTS.md) | 控制变量、HPO、训练规则与质量验收 |
| [评估契约](contracts/EVALUATION.md) | 选择指标、阈值、区间和冻结条件 |
| [产物契约](contracts/ARTIFACTS.md) | Run、队列、失败和盲测留痕 |

## 最终结果与正式报告

| 批次 | 归档与报告入口 |
| --- | --- |
| Batch A（47 Run） | [研究归档](finish-task/batch-a/summary.md) · [模型结果索引](finish-task/batch-a/model-index.json) · [主报告](../reports/batch-a/255a9bfe53ee/report.md) · [LaTeX](../reports/batch-a/255a9bfe53ee/report.tex) |
| Batch A 补充实验（S1 / S2） | [S1 验证报告](../reports/batch-a-s1/report.md) · [S2 收敛边界报告](../reports/batch-a-s2/report.md) |
| Batch B Encoder（12 Run） | [S2 验收](finish-task/batch-b-s2.md) · [正式测试验收](finish-task/batch-b-final.md) · [测试报告](../reports/batch-b-final/2605f6766085/report.md) |
| Batch B S3 / S4 Qwen（3 Run） | [S3 性能对照](../reports/batch-b-s3/performance/report.md) · [S4 性能报告](../reports/batch-b-s4/performance/report.md) · [S4 完成归档](finish-task/batch-b-s4.md) · [S4 正式测试报告](../reports/batch-b-s4/formal/final-test.md) |
| 前端与推理应用 | [前端交接说明](finish-task/frontend-freeze.md) · [评测数据快照](../apps/web/src/data/dashboard-data.json) · [导出脚本](../scripts/export_web_data.py) |

## 历史过程归档（`archive/`）

过程性阶段验收、批次执行计划与运行手册已归档至 `docs/archive/`，保留原始审计口径与运行参数供论文撰写核对：

- **阶段验收**：[Stage 0 数据审计](archive/stages/stage0-data-audit.md) · [Stage 2 冒烟](archive/stages/stage2-smoke.md) · [Stage 3 训练](archive/stages/stage3-training.md) · [Stage 4 冻结审计](archive/stages/stage4-final-evaluation.md) · [Stage 5 交付](archive/stages/stage5-delivery.md)
- **执行计划**：[Batch A 执行计划](archive/plans/2026-11-17-batch-a-execution.md) · [Batch B 总览](archive/plans/batch-b/overview.md) · [S1 契约](archive/plans/batch-b/s1-encoders.md) · [S2 复验](archive/plans/batch-b/s2-followups.md) · [S3 Qwen 提案](archive/plans/batch-b/s3-qwen-proposal.md) · [资源清单](archive/plans/batch-b/model-resources.md) · [正式测试规划](archive/plans/batch-b/final-test.md)
- **运行手册**：[Batch A 进度](archive/runbooks/batch-a-progress.md) · [Batch A S1](archive/runbooks/batch-a-s1.md) · [Batch A S2](archive/runbooks/batch-a-s2.md) · [Batch B S1](archive/runbooks/batch-b-s1.md) · [Batch B S2](archive/runbooks/batch-b-s2.md) · [Batch B S3 预检](archive/runbooks/batch-b-s3-preflight.md) · [Batch B S4 正式训练](archive/runbooks/batch-b-s4-formal.md)

原始数据审计存放于本地 `data/processed/`，训练和冻结证据保存在 `artifacts/`。
