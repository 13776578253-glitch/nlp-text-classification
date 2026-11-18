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
| [Batch A 研究归档](finish-task/batch-a/summary.md) · [模型结果索引](finish-task/batch-a/model-index.json) | 冻结测试、两轮验证补充实验与原始证据 |
| [Batch A Supplement S1](runbooks/batch-a-s1.md) | GoEmotions 统一 24 轮预算、收敛诊断和多种子复验 |
| [Batch A Supplement S2](runbooks/batch-a-s2.md) | GoEmotions 候选筛选、36 轮续训与恢复审计 |
| [Batch B 规划](plans/batch-b/overview.md) · [S1 契约](plans/batch-b/s1-encoders.md) | Encoder、补充实验和 Qwen 提案 |
| [Batch B S1 运行与验收](runbooks/batch-b-s1.md) | 资源、Tokenizer、CUDA 预检和训练入口 |
| [Batch B S1 验证报告](../reports/batch-b-s1/report.md) · [结果索引](../reports/batch-b-s1/results.json) | 八个训练槽位、选模、训练曲线与 S2 评估入口 |
| [Batch B S2 运行手册](runbooks/batch-b-s2.md) | 三种子复验、B1 预算对照与拟冻结结果 |
| [Batch B S2 验收](finish-task/batch-b-s2.md) · [研究报告](../reports/batch-b-s2/report.md) | 三种子统计、低支持标签分析与拟冻结清单 |
| [Batch B 正式测试验收](finish-task/batch-b-final.md) · [测试报告](../reports/batch-b-final/2605f6766085/report.md) | 12 个冻结 Run、逐样本复算、配对比较与中文评价 |
| [Batch B S3 性能对照](../reports/batch-b-s3/performance/report.md) · [预检运行手册](runbooks/batch-b-s3-preflight.md) | Qwen3.5 QLoRA 的 Windows/WSL2 计时、CUDA 内核与恢复验证 |
| [当前状态](current-task/ACTIVE.md) · [下一步](NEXT.md) | 阶段与证据引用、当前工作入口 |
| [待议与研究局限](BACKLOG.md) | 后续工作与研究边界 |

长期约束更新到各契约。状态页引用原始证据，原始数据审计存放于本地 `data/processed/`，训练和冻结证据保存在 `artifacts/`。Stage 5 从冻结产物生成实验结果。
