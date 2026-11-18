# Batch B Encoder 正式测试交付验收

2026-11-18 完成 12/12 项独立冻结审计及 12/12 项正式测试。清单版本为 `2605f6766085ab675d97de6a48003f1a38ee78174807d0c2e0bc8861b9cbd6f1`。四个 seed 42 主 Run 来自 S1，八个复验来自 S2。B1 采用 S2 验证预算决策选出的六轮配置。 [S2 验证与预算](batch-b-s2.md) · [正式冻结清单](../../reports/batch-b-final/2605f6766085/freeze.json)

## 冻结与复算

- 冻结前核对数据版本、预训练模型 Revision、Tokenizer、标签序列、配置、已选检查点与阈值的文件指纹。在 RTX 4070 Ti SUPER 上严格重载真实检查点，65,631 条验证记录的 float32 分数逐项精确重现，阈值候选和离散预测逐项一致。 [独立冻结审计](../../reports/batch-b-final/2605f6766085/freeze-audit.json)
- B4 三个检查点均为 `BertForSequenceClassification` 两维分类头，训练损失使用交叉熵，评分使用第二列 Softmax 概率。原始标签 0 映射到正类 1，原始标签 1 映射到负类 0。 [评估契约](../plans/batch-b/final-test.md)
- 正式测试入口先验证内容寻址清单和全部冻结文件，随后一次执行 12 个 Run。逐样本复算核对 49,950 条记录的样本 ID、原始标签、编码目标、概率与冻结阈值预测。用 scikit-learn 独立复算 F1、逐类 P/R/F1/support、Accuracy、Average Precision、中文 ROC-AUC 与混淆矩阵，以及英文 Hamming Loss 与 Samples-F1。 [验收指纹](../../reports/batch-b-final/2605f6766085/acceptance.json) · [独立测试复算日志](../../artifacts/batch-b-final/independent-test-replay.log)
- 报告的六份文件重建后逐字节一致。Batch A 检查 47 个原正式测试 Run 及 18 份报告文件，复核两轮补充实验 24 个和 4 个 Run；Batch B S1/S2 分别复核原 8 个和 9 个 Run，包括 B1 的长预算探针。37 项 pytest、Ruff 和 Git 空白检查通过。 [既有证据验收](../../artifacts/batch-b-final/prior-evidence-audit.json)

## 结果解读

- GoEmotions 三种子测试 Macro-F1 的排序在三个 seed 上一致，B3 > B2 > B1。B3−B2 的固定三种子平均差为 +0.0220，成对 95% 区间 [+0.0106, +0.0335]。Micro-F1 平均差为 +0.0106，95% 区间 [+0.0065, +0.0149]。Subset Accuracy 在 seed 123 由 B2 领先，固定三种子平均差区间跨零。 [配对结果与逐种子分数](../../reports/batch-b-final/2605f6766085/report.md)
- B4 的种子 42 测试 Macro-F1 为 0.8916，三种子均值为 0.9088，样本标准差为 0.0149。正类 Recall 在三个种子上均高于 Precision。种子 42 有 23 个假正例、17 个假负例。三种子 ROC-AUC 为 0.9655 ± 0.0024。 [中文评价与混淆矩阵](../../reports/batch-b-final/2605f6766085/report.md#b4-中文电影评论)

## 证据路径

每个 Run 的预测、指标和验证过的测试清单位于 `artifacts/batch-b-final/test/<freeze_sha256>/<model>-<seed>/`。正式输出的版本、12 个 Run 与本地证据路径见 [机器可读结果](../../reports/batch-b-final/2605f6766085/results.json)。进度与命令日志位于 [批次进度](../../artifacts/batch-b-final/progress.md)、[冻结日志](../../artifacts/batch-b-final/audit-terminal.log)、[测试日志](../../artifacts/batch-b-final/test-terminal.log) 和 [报告重建日志](../../artifacts/batch-b-final/report-rebuild-terminal.log)。评估器和预设协议提交为 `1265814`。S3 处于 [独立规划](../plans/batch-b/s3-qwen-proposal.md)。
