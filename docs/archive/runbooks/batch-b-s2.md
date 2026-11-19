# Batch B S2 · 执行与验收

[固定矩阵与选模规则](../plans/batch-b/s2-followups.md) · [S2 配置](../../configs/experiment_suites/batch_b_s2.yaml)

```powershell
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-b-s2 audit
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-b-s2 run
.venv/Scripts/python.exe -m nlp_text_classification.cli batch-b-s2 status
.venv/Scripts/python.exe -m nlp_text_classification.cli batch-b-s2 report
```

`audit` 核对 S1 的报告、八个 Run 和离线资源，验证训练器、数据和 Tokenizer 的指纹。通过后在同一 CUDA 环境串行运行九个 S2 槽位。先为 B1 在 12 轮协议下训练 seed 42，记录预算选择，再复验四个模型。各 S2 槽位从官方权重独立训练。中断后用同一命令恢复该槽位或复用已验收结果。

`artifacts/batch-b-s2/progress.md` 为实时快照，`progress.jsonl` 记录训练、验证、重载、预测和验收过程。`queue.json` 索引新 Run、配置、权重、阈值、预测和日志。B1 的预算判据及证据绑定在 `b1-selection.json`。S2 报告通过 `report` 从同一证据重建。
