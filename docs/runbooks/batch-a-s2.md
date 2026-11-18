# Batch A Supplement S2 · GoEmotions 收敛边界

S1 使用统一预算完成六种模型的验证对照。S2 根据三种子 S1 轨迹定向续训，用于定位入选模型的验证平台期。Batch B 的 E4 Transformer 对照按其独立矩阵执行。

[收敛报告与曲线](../../reports/batch-a-s2/report.md) · [筛选和逐 Run 数据](../../reports/batch-a-s2/results.json) · [S2 协议](../../configs/experiment_suites/batch_a_s2.yaml)

## 运行与复现

```powershell
.venv/Scripts/python.exe -m nlp_text_classification.cli batch-a-s2 screen
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-a-s2 run
.venv/Scripts/python.exe -m nlp_text_classification.cli batch-a-s2 report
```

筛选读取已验收的 18 条 S1 轨迹。至少一个种子的最佳轮次在第 21–24 轮，且末四轮验证 Macro-F1 斜率为正、均值超过前四轮时，配置入选。续训沿用各自数据、词表、配置、阈值网格与 seed，预算上限为 36 轮、patience 6。Seed 42 提升达到 0.01 时追加另两个种子。

续训优先读取 S1 最新完整状态，其中包括 Adam、模型、采样器及 CPU/CUDA RNG，学习率固定。恢复文件缺失时，可独立重放 S1 轨迹并核对已选检查点。报告由审计通过的队列与逐轮数据重建。

## 交付验收

- S1 筛选覆盖 18 个 Run。Vanilla RNN 有三个种子满足入选条件，LSTM 的 seed 42 满足。BiLSTM 三个种子都在第 13–15 轮达到最佳，随后触发早停。
- 两个 seed 42 续训和两个 Vanilla RNN 追加种子通过审计。四个 Run 都从 S1 第 24 轮恢复，继承的曲线精确一致。
- Vanilla RNN 三个 seed 的最佳点在第 35–36 轮，末段仍呈上升趋势。以定位平台期为目标时，只有该模型需要另行增加预算。LSTM 的 seed 42 在第 30 轮早停，最佳仍是第 24 轮。
- S2 的 10 个报告产物重建逐字节一致，26 项 pytest 和 Ruff 通过。核对了全部 24 个 S1 运行的文件指纹，以及 Batch A 原冻结清单、47 份正式测试和 18 个报告文件。

本轮训练引用代码提交 `bc4e2a4`。`artifacts/batch-a-s2/` 中的 `screening.json`、`primary.json`、`queue.json`、`results.json` 和 `delivery_audit.json` 分别记录候选依据、复验门槛、运行指纹、恢复关系、最终验收。逐轮日志和检查点随各 Run 归档。
