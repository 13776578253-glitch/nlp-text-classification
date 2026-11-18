# Batch B S1 · Encoder 实验契约

| 配置 | 预训练仓库 | 任务 | 基础 LR | 候选 LR | 初始微批次 |
| --- | --- | --- | ---: | ---: | ---: |
| B1 | distilbert/distilbert-base-uncased | GoEmotions | 2e-5 | 3e-5 | 32 |
| B2 | answerdotai/ModernBERT-base | GoEmotions | 2e-5 | 3e-5 | 16 |
| B3 | answerdotai/ModernBERT-large | GoEmotions | 2e-5 | 1e-5 | 8 |
| B4 | hfl/chinese-macbert-base | 中文电影评论 | 2e-5 | 3e-5 | 16 |

所有参数完整微调。每个槽位使用 seed 42、AdamW、weight decay 0.01、线性衰减、6% warmup、有效批次 32、6 epoch 上限和 patience 2。末尾不足一批时按实际样本数归一化梯度。全局梯度范数上限 1.0。四个基础槽位先完成，四个候选槽位随后执行，RTX 4070 Ti SUPER 单卡串行。验证成绩平分时保留更早轮次，候选平分时保留基础配置。每个槽位只对确认的资源故障执行一次可恢复调整，微批减半，梯度累积保持有效批次。预检确定的资源设置在同模型两个槽位对齐。任何后续差异单列为 S2 复验依据。

## 数据、预测与长度

GoEmotions 用 28 维标签顺序训练 BCEWithLogitsLoss，评分为 float32 sigmoid。中文使用两个 logit 和 cross-entropy，评分取 float32 softmax 的第二列（零基索引 1）。语义 class 1 对应原始标签 0（正向），class 0 对应原始标签 1（负向）。两种分数共用 0.05 到 0.95 的概率阈值网格，预测规则为 `score >= cutoff`。见 [正式冻结与测试契约](final-test.md)。

英语 Tokenizer 接收原始冻结文本。中文保留分词边界，用单个 ASCII 空格连接原有词元。审计在训练和验证分区对比删空格后的 Token ID 变化，特别覆盖混合英文和数字的样本。计算长度时包含特殊 Token。每个模型选使训练覆盖率达到 99% 的最小候选长度。英语候选为 64、96、128，中文为 128、192、256。全部低于 99% 时用最大候选，并记录实际截断比例。右侧截断与动态右侧 Padding，向上对齐 8。逐模型审计 train/validation 长度分位数、截断比例、特殊 Token 和 Mask。

## 门禁与留档

- 资源阶段锁定 Hugging Face Commit SHA、许可证、Tokenizer 文件、权重文件和哈希。离线加载使用标准模型类、`trust_remote_code=False`。优先 safetensors，官方只有 PyTorch 权重时通过 PyTorch 2.6 以上的 `weights_only=True` 路径加载。
- 每个模型先完成真实小样本的训练、反传、最优权重重载和验证分数复算。显存不足时只调整微批与累积或开启梯度检查点，并记录实际设置和峰值。
- CUDA 支持 BF16 时采用 BF16 autocast，权重和优化器保留 FP32。GPU 锁沿用 Batch A，训练进程记录学习率、更新次数、损失、验证指标、显存、停止原因、最佳模型和逐轮完整恢复检查点。
- 训练入口只加载 `training.json` 指向的 train/validation。每个候选保存逐样本训练和验证分数、指标、阈值网格和选模记录。独立验收复算 Macro-F1、早停和候选排名，核对 checkpoint、数据、Tokenizer 与代码指纹。验证推理吞吐以同一设备、模型各自预检通过的评估批次计，并记录批次、序列长度策略和样本数。
- 正式 S1 冻结有限搜索预算后只进行训练和验证。报告以验证集对照 Batch A，输出量化的收敛、异常、稳定性或资源复验建议。性能分数作为结果展示。

[运行矩阵](../../../configs/experiment_suites/batch_b_s1.yaml) · [Batch A 评估契约](../../contracts/EVALUATION.md)
