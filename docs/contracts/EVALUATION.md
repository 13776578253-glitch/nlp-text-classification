# Batch A 评估与冻结契约

| 数据集 | 验证选择 | 最终指标 |
| --- | --- | --- |
| GoEmotions | 全局阈值和最佳 epoch 以 validation Micro-F1 为准。 | Micro-F1、Macro-F1、Samples-F1、Hamming Loss、Subset Accuracy、逐标签 P/R/F1/support，以及适用的 PR-AUC。 |
| 中文电影评论 | 建议按 validation Macro-F1 选择阈值和最佳 epoch。 | Accuracy、Macro-F1、语义正类 P/R/F1、混淆矩阵、ROC-AUC/PR-AUC，另附基于逐样本预测的置信区间。 |

分数要标明 SVM decision margin 或神经网络 sigmoid。分数、比较方向、阈值候选、并列处理、最佳 epoch 选择和缺失指标规则由配置记录，只访问 validation 做选择。中文正类指标按冻结的语义映射计算。两个数据集各自做模型排序和对照。

1. Stage 0 审计 test 的格式、哈希和分区隔离，留存数据证据。Stage 3 运行时只有 train/validation 路径和必要的训练数据指纹。
2. Stage 4 先独立核对 14 个成功 Run 的数据、配置、词表、最佳权重、阈值及报告计算版本。每个矩阵键对应唯一 Run。冻结记录存放于该批次的 `final_runs.yaml`，一次选齐两个数据集。
3. 评估入口先检查冻结覆盖率、各指纹和重复执行状态，再读取 test 并计算冻结方案的输出。写入的预测和指标使用原子发布且禁止覆盖。中途故障留下状态和证据，恢复只能继续缺失项，使用原冻结输入。

现有单 Run 的最终评估入口已下线。批次冻结验证器和正式 test 执行器在后续阶段实现。
