# Batch A Supplement S1 · GoEmotions 延长训练

S1 使用与原 Batch A 相同的六个英文神经模型配置，在统一的 24 轮上限和 patience 6 下比较模型对训练预算的敏感度。E1 使用原验证成绩作为排名参照，E2 与两个 E5 条件继续执行受控消融。

[收敛报告与六张曲线](../../reports/batch-a-s1/report.md) · [逐模型与种子数据](../../reports/batch-a-s1/results.json) · [S1 协议](../../configs/experiment_suites/batch_a_s1.yaml)

## 运行与复现

```powershell
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-a-s1 primary
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-a-s1 replicate
.venv/Scripts/python.exe -m nlp_text_classification.cli batch-a-s1 report
```

第一条命令先在 seed 42 下独立重放各模型前 12 轮，再从验证通过的 S1 检查点接续。提升达到 0.02、最佳轮次在第 22–24 轮或任何参与比较的模型对发生排名互换时触发复验。TextCNN 三个条件共用复验名单。发布触发名单后，第二条命令补跑 seed 123、2026。第三条命令核验已有运行和文件指纹，从现有证据重建报告。

## 交付验收

- 6 个前缀重放、6 个 seed 42 延长结果、12 个复验全部通过。第 12 轮的模型、Adam、CPU/CUDA/采样器 RNG 以及获选检查点逐项一致。有原始验证复验的 8 个追加 Run 也通过了共同前缀核对。
- RNN 和 LSTM 三个种子的提升均达到 0.02。RNN 三个种子的最佳轮次为 23、24、24。BiLSTM 与 E5-TUNE 的排名互换在 seed 42 和 2026 出现。报告建议为英文神经模型另立长预算协议。
- 原冻结清单、47 份正式测试和原报告的 18 个输出均通过前后 SHA-256 核对。S1 的 8 个报告产物重建逐字节一致，26 项 pytest 和 Ruff 通过。
- 训练和复验统一引用代码提交 `c971cac`。复验记录的工作区改动来自待提交的 S1 首轮报告，源码与配置已对该提交核对。

原始证据在 `artifacts/batch-a-s1/`。`queue.json` 索引 Run 和文件指纹，`prefix_audit.json` 记录完整状态比对，`primary.json` 保存首轮复验决策，`base_preservation.json` 和 `delivery_audit.json` 保存原批次保护与交付复核记录。
