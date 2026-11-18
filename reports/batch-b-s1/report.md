# EmotionLab · Batch B S1 验证报告

四个官方 Encoder 以 seed 42 完整微调，每个模型比较一个基础配置和一个学习率候选。本报告使用冻结 v1 的训练与验证分区。Batch A 参照来自原冻结清单的验证分数。

## 入选结果

| 配置与曲线 | 模型 | 数据集 | 验证 Macro-F1 | Batch A E1 验证 F1 | 差值 | 最佳/实际 Epoch | 参数量 | 推理样本/秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [B1](curves/B1.png) | distilbert/distilbert-base-uncased | goemotions | 0.4958 | 0.4778 | +0.0181 | 5/6 | 66,975,004 | 4215 |
| [B2](curves/B2.png) | answerdotai/ModernBERT-base | goemotions | 0.5305 | 0.4778 | +0.0527 | 2/4 | 149,626,396 | 1712 |
| [B3](curves/B3.png) | answerdotai/ModernBERT-large | goemotions | 0.5285 | 0.4778 | +0.0507 | 2/4 | 395,859,996 | 870 |
| [B4](curves/B4.png) | hfl/chinese-macbert-base | movie_reviews_zh | 0.9076 | 0.8581 | +0.0496 | 2/4 | 102,269,186 | 1527 |

## 全部训练槽位

| 槽位 | Run ID | LR | 验证 F1 | 阈值 | 最佳/实际 Epoch | 停止原因 | 微批次 | 有效批次 | 分钟 | 峰值 allocated GiB | 峰值 reserved GiB | 推理批次 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline/B1 | 20261118T015803-9a734c2ff1 | 2e-05 | 0.4852 | 0.20 | 6/6 | epoch_cap | 32 | 32 | 3.41 | 1.34 | 1.52 | 32 |
| baseline/B2 | 20261118T020131-dff5251420 | 2e-05 | 0.5177 | 0.20 | 4/6 | early_stopping | 16 | 32 | 14.72 | 3.19 | 3.33 | 32 |
| baseline/B3 | 20261118T021618-7daa0e9e37 | 2e-05 | 0.5285 | 0.30 | 2/4 | early_stopping | 16 | 32 | 19.72 | 7.91 | 8.46 | 32 |
| baseline/B4 | 20261118T023608-f5ee732530 | 2e-05 | 0.9016 | 0.75 | 4/6 | early_stopping | 16 | 32 | 6.60 | 2.66 | 2.77 | 32 |
| alternate/B1 | 20261118T024246-45479325d9 | 3e-05 | 0.4958 | 0.25 | 5/6 | epoch_cap | 32 | 32 | 4.07 | 1.34 | 1.56 | 32 |
| alternate/B2 | 20261118T024653-ae365ea198 | 3e-05 | 0.5305 | 0.25 | 2/4 | early_stopping | 16 | 32 | 10.58 | 3.19 | 3.33 | 32 |
| alternate/B3 | 20261118T025731-1ab1a13e9e | 1e-05 | 0.5215 | 0.15 | 2/4 | early_stopping | 16 | 32 | 19.39 | 7.91 | 8.46 | 32 |
| alternate/B4 | 20261118T031702-dd2ab25555 | 3e-05 | 0.9076 | 0.60 | 2/4 | early_stopping | 16 | 32 | 4.88 | 2.66 | 2.77 | 32 |

## Tokenization 与证据

- B1 · 64 token，训练/验证截断比例 0.01% / 0.00%。
- B2 · 64 token，训练/验证截断比例 0.03% / 0.04%。
- B3 · 64 token，训练/验证截断比例 0.03% / 0.04%。
- B4 · 128 token，训练/验证截断比例 0.56% / 0.71%。

完整指标和数据、模型、Tokenizer 指纹见 `results.json`。训练状态、逐样本预测和审计见 `artifacts/batch-b-s1/formal/`。

## S2 评估入口

- B1 · 最佳轮次 5/6。固定已选配置做单模型延长，再按验证趋势决定复验。
- B2 / B3 · 前两名验证差 0.0020。使用原入选超参数和固定追加种子复验排序稳定性。

S2 实际矩阵待评估。正式测试与 S3 为独立阶段。
