# Batch B S3 · Qwen 分类器提案

已批准配置为 Qwen3.5-4B-Base、248 个线性投影的 NF4 QLoRA（Rank 32 / Alpha 64）和 GoEmotions 28 维原生分类头。研究问题是通用 LLM 在同一 28 标签任务上的验证表现与训练成本。

S3 已完成权重加载、Tokenizer 与分类任务预检，并复验 WSL2 FLA/causal-conv1d 的训练和独立进程恢复。正式实验由 S4 接续。 [S3 配置](../../../configs/experiment_suites/batch_b_s3.yaml) · [工程入口](../../runbooks/batch-b-s3-preflight.md) · [性能对照](../../../reports/batch-b-s3/performance/report.md) · [S4 正式测试](../../../reports/batch-b-s4/formal/final-test.md)
