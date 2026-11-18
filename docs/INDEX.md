# 项目文档

| 文档 | 职责 |
| --- | --- |
| [项目范围](PROJECT_SPEC.md) | 课程范围、研究问题和批次边界 |
| [架构](ARCHITECTURE.md) | 目录、模块职责与执行依赖 |
| [数据契约](contracts/DATASETS.md) | 来源、分区、标签和预处理边界 |
| [实验矩阵](contracts/EXPERIMENT_MATRIX.md) | Batch A/B 范围、主结果位、消融复用和复验矩阵 |
| [实验契约](contracts/EXPERIMENTS.md) | 控制变量、HPO、训练规则与质量验收 |
| [评估契约](contracts/EVALUATION.md) | 选择指标、阈值、区间和冻结条件 |
| [产物契约](contracts/ARTIFACTS.md) | Run、队列、失败和盲测留痕 |
| [执行计划](plans/2026-11-17-batch-a-execution.md) | 阶段依赖、预算与验收条件 |
| [Batch A Supplement S1](runbooks/batch-a-s1.md) | GoEmotions 统一 24 轮预算、收敛诊断和多种子复验 |
| [当前状态](current-task/ACTIVE.md) · [下一步](NEXT.md) | 阶段与证据引用、当前工作入口 |

长期约束更新到各契约。状态页引用原始证据，原始数据审计存放于本地 `data/processed/`，训练和冻结证据保存在 `artifacts/`。Stage 5 从冻结产物生成实验结果。
