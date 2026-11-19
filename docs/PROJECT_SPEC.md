# 课程项目范围 · Batch A V2

课程选题 4 要求建立文本分类系统，并在给定或公开数据上比较方法。Batch A 以 GoEmotions simplified 的英文 28 标签多标签分类和中文电影评论二分类作为两套正式任务。

- RQ1 比较同一数据集上 TF-IDF + LinearSVC、TextCNN、Vanilla RNN、LSTM 和 BiLSTM。
- RQ2 在同一 TextCNN 中比较随机词嵌入、Word2Vec 微调和 Word2Vec 冻结。随机组引用 E2，每套数据合计 7 个配置。
- RQ3 比较语言适配的预训练 Transformer Encoder。Batch B S1 包括 DistilBERT、ModernBERT-base/large 和 Chinese MacBERT-base。两套任务分别比较，沿用 Batch A v1 数据。 [Batch B 规划](plans/batch-b/overview.md)

Batch A Stage 5 交付模型对照、调参记录、多种子统计、消融分析、复现手册、验收证据和 LaTeX 初稿。Batch B S3 / S4 的 QLoRA 分类器采用独立预研、验证冻结和正式测试。GPU 极限、20 Newsgroups 与复杂前端开发列为独立工作。 [S4 完成归档](finish-task/batch-b-s4.md)

[执行计划](plans/2026-11-17-batch-a-execution.md) · [数据契约](contracts/DATASETS.md) · [实验矩阵](contracts/EXPERIMENT_MATRIX.md)
