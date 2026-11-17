# Batch A 实验契约

[实验矩阵](EXPERIMENT_MATRIX.md) 定义主结果位和复验对象。[批次配置](../../configs/experiment_suites/batch_a.yaml) 提供参数、预算和阈值。模型选择、阈值与统计见 [评估契约](EVALUATION.md)。

## 基础配置与受控变量

- E1 使用 TF-IDF + LinearSVC，英文 OvR，中文二分类 margin。统一用对应 train 拟合 TF-IDF，以验证 Macro-F1 选参数与阈值。基线 C=1、1–2 gram、min_df=2、max_features=100,000、sublinear TF、balanced class weight。
- 神经模型使用相同数据分区、同任务词表、50d Embedding、统一训练上限 12 epoch、patience 3、有效 batch size 64、grad clip 1.0、Adam。基线学习率 0.001、weight decay 0、dropout 0.3。固定 TextCNN 卷积核 3/4/5 各 100 filters，循环模型 hidden size 128。长度候选 128/256 由 Stage 0 的 train 分布确认。
- E2、E5-TUNE、E5-FROZEN 在同数据集、同 seed 下共用同一 TextCNN、词表、数据版本、参数预算、非 Embedding 初始化和数据顺序。使用独立的参数、数据和训练 RNG 流，避免词向量分支改变其余层初始状态。保存非 Embedding 初始化指纹，同一个 seed 的 E2/E5 必须一致。
- E5-TUNE 与 E5-FROZEN 共用初始 Word2Vec 矩阵，词表未命中项按同一个 seed 初始化，PAD 行规则一致。英文 Word2Vec 按冻结 train 版本、seed 42 预训练一次，正式训练和复验共用其指纹。中文各 Run 共用同一原始向量版本。报告训练词表的类型覆盖率与训练词频加权覆盖率，分母去除 PAD/UNK 等保留项。记录词向量来源、哈希、初始化哈希和 OOV 策略。每个冻结条件自动核对无梯度、优化器排除，以及初始、最终和获选 checkpoint 的 Embedding 字节一致。

## Stage 3 · Optuna 搜索

先按矩阵完成 14 个 seed 42 基础配置，然后按数据集、模型建独立的 Optuna Study。候选预算在基线之外。使用 TPE sampler，采样 seed 42。SVM 和神经 Study 分别预留 5 和 4 个 startup Trial。完整 Validation Macro-F1 最高的可验收候选获选，基础配置可以获选。受控消融按 E2 获选参数补跑 E5 两组，完全匹配的基础 Run 可直接引用。E5 独立优化另设 Study，当前预算为 0，产物不得进入受控消融或 14 个主结果位。

| 范围 / 预算 | SVM | E2 | E3-RNN | E3-LSTM / E3-BiLSTM |
| --- | --- | --- | --- | --- |
| 每数据集每模型候选 | 16 | 6 | 6 | 各 6 |
| 对数采样 | C 0.01–10 | LR 0.0002–0.003 | LR 0.0001–0.002 | LR 0.0001–0.003 |
| 离散取值 | min_df 2/3/5，1-gram 或 1–2 gram，class weight null 或 balanced | weight decay 0 / 1e-5 / 1e-4 / 1e-3，dropout 0.1 / 0.3 / 0.5 | 同 E2 | 同 E2 |

- SVM 一次拟合。神经模型每 epoch 报告同一验证指标，三轮无改进早停。MedianPruner 在至少 4 个通过质量检查的完整 Trial 后，最早在第 6 个 epoch 结束时检查，此后每两轮检查。正常早停且通过验收计作完整。正常剪枝耗用候选槽，不参加获选。数值与数据错误进入 FAIL。
- 神经模型的 6 个 Trial 是初始预算。若完整并通过质量检查的候选仍少于 4 个，每个 Study 最多追加 2 个关闭剪枝的 Trial，补足即停止。基线不计入 4 个完整 Trial，补充后仍不足时记录不足和原因，暂停该主结果位放行。
- 失败候选保留其槽位。可恢复故障最多重试一次，同一候选参数关联新 Run，消耗独立的重试槽。每数据集/模型的候选尝试、剪枝、完整 trial、失败与重试数同时披露。SVM 与神经模型分别展示预算、有效 epoch、实际耗时和比较边界。
- 每个 Study 结束时核对候选槽位都已处理，且至少有一个额外 Trial 完成并通过质量检查。Study 可选基础配置作为最终结果。候选缺失或整组失败时记录实际完成度与阻塞证据。
- 所有搜索范围在 Stage 0 版本化。调参只访问 train/validation。改变影响实验协议的范围、优化器、预算或训练数据时记录新配置版本和受影响的可比组，按新版本执行并保留旧 Run。
- 每个设备按预定队列运行，GPU 独占串行。可恢复 OOM 先减微批次并增加累积步数，保持有效 batch 与更新次数。E2/E5 受影响时同步对齐三组的微批次/累积规则，无法证实等价的组按新配置重跑并计入批准预算。

## 自动训练验收

| 检查 | 放行证据 |
| --- | --- |
| 输入、目标和损失 | 同任务分区和预处理指纹一致；GoEmotions 目标和 logit 为 B×28，中文为 B×1。SVM 对应 28 个 OvR margin 或 1 个有固定符号的二分类 margin。 |
| 数值、学习与停止 | 损失、参数和梯度有限，预期更新的参数块在有效训练步中有梯度并至少一次有效更新。记录 loss 轨迹、每 epoch 验证指标和停止原因，按日志重算 Early Stopping / Pruning 决策。常数预测、全空预测、持续无有效更新等进入诊断；损失可波动。 |
| Checkpoint | 最佳 checkpoint 能在同设备评估模式重载。分数误差满足 atol 1e-7、rtol 1e-6，离散预测严格一致。验证集分数、阈值和指标与获选记录一致。SVM 同样验证序列化、分数及预测。 |
| 分区隔离与消融 | 训练在没有 test 路径或无权读取 test 时通过，训练拟合器只接收 train；所有 Run 引用版本化分区。E2/E5 初始化、预处理与参数约束通过，冻结 Embedding 字节一致。 |
| 可重算 | 从全精度原始预测重算全部已定义的验证指标，容差 1e-12，逐类计数和混淆矩阵逐项一致。 |

正式候选须全项通过。失败时先诊断再按预算重试，保留原始日志。复验规则在 [实验矩阵](EXPERIMENT_MATRIX.md) 与 [评估契约](EVALUATION.md) 中定义。

## PERFORMANCE_REVIEW

质量通过但训练 Macro-F1 低于 0.25 且持续偏低、验证预测单一的比例达到 0.99 或输出标签覆盖低于 0.25、三种子 Macro-F1 样本标准差达到 0.02 时，记录训练/验证差距、loss、停止原因、标签覆盖、支持度和种子分布。多标签的空预测只在覆盖及 Macro-F1 同时异常时解释为退化。性能诊断与技术 FAIL 分开记录。

每个数据集最多追加一个验证诊断 Run，优先处理退化、欠拟合、再处理高方差。对神经模型可比较 18 epoch 上限下的训练曲线；高方差优先复核同配置追加种子；SVM 可检查已冻结搜索空间内尚未覆盖的配置。诊断仅使用训练和验证集。E5 的补充诊断单列，正式受控组保持原冻结关系。诊断结果与原 Run、补充理由和限制同存，进入报告的性能说明，主表依已确定的选取规则产出。

[Optuna TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) · [Optuna MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html)
