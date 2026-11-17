# 课程项目范围 · Batch A V2

课程选题 4 要求建立文本分类系统，并在给定或公开数据上比较方法。Batch A 以 GoEmotions simplified 的英文 28 标签多标签分类和中文电影评论二分类作为两套正式任务。

- RQ1 比较同一数据集上 TF-IDF + LinearSVC、TextCNN、Vanilla RNN、LSTM 和 BiLSTM。
- RQ2 在同一 TextCNN 中比较随机词嵌入、Word2Vec 微调和 Word2Vec 冻结。随机组引用 E2，每套数据合计 7 个配置。
- Batch B 再做语言适配的 Transformer 对照。英文 DistilBERT / ModernBERT 草案位于 `configs/batch_b/`，中文候选待 Batch B 核验。20 Newsgroups 和 GPU 极限实验移出本次课程实验。

Stage 5 从真实的冻结结果生成两套数据的对照、消融分析、复现手册和验收证据。正式测试完成后再决定前端工作，课程报告和生成文件随真实结果落地。

[执行阶段与当前状态](plans/2026-11-17-batch-a-execution.md) · [数据契约](contracts/DATASETS.md) · [实验契约](contracts/EXPERIMENTS.md)
