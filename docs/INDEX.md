# 项目文档

| 文档 | 职责 |
| --- | --- |
| [项目范围](PROJECT_SPEC.md) | 课程范围、研究问题和批次边界 |
| [架构](ARCHITECTURE.md) | 目录、模块职责与执行依赖 |
| [数据契约](contracts/DATASETS.md) | 来源、分区、标签和预处理边界 |
| [实验契约](contracts/EXPERIMENTS.md) | 模型矩阵、控制变量和训练规则 |
| [评估契约](contracts/EVALUATION.md) | 选择指标、阈值和冻结条件 |
| [产物契约](contracts/ARTIFACTS.md) | Run、队列、失败和盲测留痕 |
| [迁移验收与执行报告](plans/2026-11-17-batch-a-execution.md) | 当前实况、阶段验收、后续实施和 Git 发布选择 |

长期约束更新到相应契约。实施中的阶段进度和实测证据记在执行报告，实验结果在 Stage 5 从冻结产物生成。
