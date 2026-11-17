# Batch A 实验契约

正式矩阵由 [批次配置](../../configs/experiment_suites/batch_a.yaml) 定义，两个数据集交叉 E1、E2、E3-RNN、E3-LSTM、E3-BiLSTM、E5-TUNE、E5-FROZEN，共 14 个首轮 seed 42 Run。E5-RANDOM 在汇总时解析为同数据集的 E2 Run。

- E1 使用 TF-IDF 与 LinearSVC。GoEmotions 使用 One-vs-Rest，中文使用二分类 margin。特征从 train 拟合。
- E2、E5-TUNE、E5-FROZEN 共享 TextCNN、训练配置、数据分区、词表、序列长度和 50d Embedding。执行时先对齐分类网络初始化，再改变 Embedding 来源或可训练性。冻结合成测试要检查无梯度及训练前后权重一致。
- E3 的三个模型共享预处理、词表、训练预算和评估实现，记录参数量，正确处理 padding 和变长序列。GoEmotions 使用 28 维 BCE，中文使用单 logit BCE。
- 矩阵中的 128/256 输入长度、12 epoch、patience 3 和有效 batch size 64 是起始配置。Stage 0 用训练分布复核长度，Stage 2 用冒烟确认形状和内存。Stage 3 的每条记录保存最终解析值、参数变更和验证选择依据。
- Stage 3 串行调度。E1 用 CPU，神经模型用经预检的同一张 GPU。首次运行和新的配置尝试分别分配 Run ID，保留原失败目录。
