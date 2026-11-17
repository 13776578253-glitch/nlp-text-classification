# Stage 5 报告交付

- [完整报告](../reports/batch-a/255a9bfe53ee/report.md)
- [LaTeX 初稿](../reports/batch-a/255a9bfe53ee/report.tex)
- [逐标签、样本数和混淆矩阵](../reports/batch-a/255a9bfe53ee/class_metrics.json)
- [质量验收与诊断记录](../reports/batch-a/255a9bfe53ee/quality_review.json)
- [复现手册](runbooks/batch-a-progress.md)

报告包含两个数据集的 14 个主结果、80 个 Optuna Trial、10 组独立种子统计、3 个追加复验、5 项中文配对 Bootstrap 对照，以及 14 张轨迹图。GoEmotions 的 6 个神经主 Run 均达到 12 epoch 上限。中文的 6 个神经主 Run 均由早停结束。逐模型的最佳 epoch 与曲线在对应结果行。

验收逐一核对了冻结清单、47 个测试清单和文件指纹、样本顺序、标签、离散决策、完整指标及独立复算的 Macro-F1。再次运行 `batch-a report`，18 份输出逐字节一致。全量 pytest 26 项通过，Ruff 通过。测试预测、模型、日志和原始数据继续作为本地证据存放在 `artifacts/` 和 `data/`。
