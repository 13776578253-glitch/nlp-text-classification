# Batch B · 正式冻结与测试契约

正式批次以 [S2 拟冻结清单](../../../reports/batch-b-s2/proposed-freeze.json) 的 12 个 Run 为输入。B1–B4 各含 seed 42 主 Run，以及 seed 123、2026 的复验。每个 Run 固定自己的最优检查点和验证阈值。冻结审计仅使用 train/validation 和已锁定的数据 Manifest，核对 S1/S2 入选配置、模型 Revision、Tokenizer、标签顺序、完整文件指纹及逐样本验证预测。从真实 checkpoint 严格重载分类模型，复验 float32 分数、离散预测及阈值。

GoEmotions 的 28 个 logit 分别应用 sigmoid。B4 是 `BertForSequenceClassification` 的两维分类头，训练损失为交叉熵。评分使用 `softmax(logits.float(), dim=1)[:, 1]`，即第二列、零基索引 1。原始标签 0 映射到语义正类 1，原始标签 1 映射到负类 0。两任务以 `score >= frozen_threshold` 生成预测。B4 复用 0.05 至 0.95 的概率阈值网格，这是 S1 契约所称的 sigmoid 阈值网格。

发布内容寻址的正式冻结清单后，测试入口首先复核冻结指纹，再读取测试分区。一次运行覆盖全部 12 个 Run；同冻结重启只补未发布项。原始 Batch A、Batch B S1/S2 证据保留，正式结果存入独立目录。

主表以 seed 42 展示验证与冻结测试分数。三种子报告逐 seed 值、均值、样本标准差（ddof=1）。中文报告 Macro-F1、Accuracy、语义正类 P/R/F1、ROC-AUC、PR-AUC、逐类与混淆矩阵。英文报告 Macro-F1、Micro-F1、Subset Accuracy 和逐标签 P/R/F1/support。

ModernBERT 配对比较以 B3−B2 为方向。按测试样本 ID 对齐真实标签和预测，整行成对有放回重采样 10,000 次，固定 RNG seed 20261118，报告 Macro-F1、Micro-F1 和 Subset Accuracy 的 95% 百分位区间。逐 seed 对照和固定三种子平均差共享同一组重采样行索引。区间反映测试样本抽样波动，种子样本标准差描述训练波动。预设分析范围见 [正式配置](../../../configs/experiment_suites/batch_b_final.yaml)。
