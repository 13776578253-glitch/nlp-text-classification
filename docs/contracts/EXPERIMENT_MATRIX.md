# 实验矩阵

[Batch A 配置](../../configs/experiment_suites/batch_a.yaml) 是机器可读的矩阵。两个数据集各展开七行，共 14 个主结果位。模型排名以各自数据集内的 validation Macro-F1 为准，各自使用固定分区、标签、预处理和测试复核。

| 实验 ID | 模型 / Embedding | GoEmotions | 中文评论 | 每数据集首轮 HPO |
| --- | --- | --- | --- | --- |
| E1 | TF-IDF + LinearSVC | OvR 多标签 | 二分类 | 16 Trial |
| E2 | TextCNN / 随机可训练 | 28 logit | 单 logit | 6 Trial |
| E3-RNN | Vanilla RNN / 随机可训练 | 28 logit | 单 logit | 6 Trial |
| E3-LSTM | LSTM / 随机可训练 | 28 logit | 单 logit | 6 Trial |
| E3-BiLSTM | BiLSTM / 随机可训练 | 28 logit | 单 logit | 6 Trial |
| E5-TUNE | TextCNN / Word2Vec 微调 | 28 logit | 单 logit | 复用同数据集 E2 入选配置 |
| E5-FROZEN | TextCNN / Word2Vec 冻结 | 28 logit | 单 logit | 复用同数据集 E2 入选配置 |

每行先完成 seed 42 基础配置，随后依 [实验契约](EXPERIMENTS.md) 运行 HPO 或受控补跑。各数据集的 E5-RANDOM 引用同 cohort、同 seed 的 E2 Run ID，在主表和消融表中共用同一结果。GoEmotions 的 Word2Vec 来源为官方 train 文本，中文为老师提供的 50d 词向量。跨数据集讨论方法的趋势和错误类型，模型排序与置信区间分别计算。

## 复验矩阵

| 每套数据的复验对象 | 种子 | 选取依据 |
| --- | --- | --- |
| E1、E2、E5-TUNE、E5-FROZEN | 42、123、2026 | 固定关键对照，独立重新训练 |
| E3-RNN / E3-LSTM / E3-BiLSTM 中最高的一种 | 42、123、2026 | 第一轮 validation Macro-F1，平分按上表顺序 |
| 可选一个三次训练区块 | 42、123、2026，或 52、62、72 | 接近时加入下一个循环模型，方差异常时针对已有模型补充种子，按预设上限执行 |

5 个必选配置 × 3 个种子 × 2 数据集＝30 个独立复验 Run。复验使用第一轮入选的超参数；每个 seed 以同一 validation 规则独立选择最佳 epoch 和阈值。首轮主结果位仍为 14 个，复验 seed 42 与主结果 seed 42 有独立 Run ID。

记录里的 `baseline` 是初始配置，`hpo` 是搜索尝试，`main` 是从合格的基础/搜索/受控 Run 引用的主结果位。三种子统计只按 `replication` cohort 的 `(dataset, experiment, seed)` 三键查询。主实验和基础训练即使恰好使用 seed 42，也不计入复验样本，额外诊断使用独立 cohort。

## Batch A Supplement S1

GoEmotions 六个神经配置使用原入选参数，在统一 24 轮上限与 patience 6 下比较验证表现。Seed 42 对照先复核原 12 轮轨迹和检查点，按固定规则触发 seed 123、2026 复验。见 [协议、结果与复现](../runbooks/batch-a-s1.md)。

## Batch A Supplement S2

从 S1 三种子验证轨迹筛选仍在上升的模型，按 36 轮上限和 patience 6 从完整训练状态续训。S2 用于检查入选模型的收敛边界，模型的统一预算比较见 S1。见 [筛选、续训与复现](../runbooks/batch-a-s2.md)。

## Batch B

E4 对应 Batch B。S1 执行四个 Encoder 的完整微调，S2 根据 S1 的运行证据制定补充矩阵，S3 为 Qwen3.5-4B 与 QLoRA 的独立提案。

| ID | 模型 | 数据集 |
| --- | --- | --- |
| B1 | DistilBERT | GoEmotions |
| B2 | ModernBERT-base | GoEmotions |
| B3 | ModernBERT-large | GoEmotions |
| B4 | Chinese MacBERT-base | 中文电影评论 |

[Batch B 总览](../plans/batch-b/overview.md) · [S1 实验契约](../plans/batch-b/s1-encoders.md)
