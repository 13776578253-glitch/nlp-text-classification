# Batch A · 研究总结与归档

Batch A 已完成数据冻结、训练实现、冒烟、正式训练、独立审计、冻结测试、报告和两轮验证补充实验。统一索引列出 14 个主实验与全部 47 个正式测试 Run，同时标记 S1、S2 的验证 Run、配置和证据目录。归档以 Git `05e277e` 为审阅起点，复核了原冻结文件、47 份测试清单、18 份原报告文件和补充 Run 的指纹。 [模型结果索引](model-index.json) · [阶段验收](../../stage5-delivery.md)

## 冻结测试集结论

- GoEmotions 上，LinearSVC 的测试 Macro-F1 为 0.4616。最佳神经主实验 TextCNN 随机组为 0.3847，Word2Vec 微调为 0.3841，冻结为 0.2569。LinearSVC 的三种子复验均为 0.4616，随机组为 0.3837 ± 0.0008。
- 中文电影评论上，主实验测试 Macro-F1 最高的是 LSTM（0.8453），LinearSVC 为 0.8371。原选模依据是验证集 Macro-F1，验证集上 LinearSVC 最高（0.8581）。TextCNN 微调与随机的测试差为 +0.0353，配对 95% 区间为 [+0.0027, +0.0705]。独立三种子测试均值分别为 0.8427 和 0.8317，后者样本标准差为 0.0210。
- 两个数据集分别回答同任务下的模型比较。中文测试集包含 369 条，跨种子波动与测试样本抽样波动分别报告；GoEmotions 的若干低支持标签测试正例很少，逐类 F1 随样本量一起解读。采用同一全局阈值和有限 HPO 预算。 [冻结结果、区间与类别分析](../../../reports/batch-a/255a9bfe53ee/report.md) · [冻结清单](../../../artifacts/batch-a/final_runs.yaml)

## 验证集补充结论

- S1 为六个英文神经模型统一放宽到 24 轮上限和 patience 6。前 12 轮完整状态重现。Vanilla RNN 与 LSTM 在三个种子上的验证增益均至少 0.02。BiLSTM 超过 E5-TUNE 的排名变化出现在 seed 42 和 2026。原 12 轮的训练预算会影响该组神经模型的验证比较。 [S1 报告](../../../reports/batch-a-s1/report.md) · [S1 协议与验收](../../runbooks/batch-a-s1.md)
- S2 定向续训 Vanilla RNN 与 LSTM。RNN 在三个种子的最佳点为第 36、36、35 轮，验证仍改善。LSTM 的 seed 42 在第 30 轮早停，最佳依然是第 24 轮。RNN 的最终平台期超出本次预算，Batch A 在 36 轮证据处收束。 [S2 报告](../../../reports/batch-a-s2/report.md) · [S2 协议与验收](../../runbooks/batch-a-s2.md)

Batch B S1 沿用双数据集 v1、标签顺序和验证 Macro-F1。初步跨批比较引用 Batch A 原验证成绩。正式测试如另行立项，按独立冻结清单选定模型、阈值和评估器。
