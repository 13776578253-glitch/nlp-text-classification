# Batch B S1 · 运行与验收

[实验契约](../plans/batch-b/s1-encoders.md) · [模型资源](../plans/batch-b/model-resources.md) · [Tokenization 锁文件](../../configs/batch_b/tokenization_lock.json)

## Tokenization 验收

| 配置 | 训练 P99 | 长度上限 | 训练截断 | 验证截断 |
| --- | ---: | ---: | ---: | ---: |
| B1 | 38 | 64 | 0.007% | 0.000% |
| B2、B3 | 37 | 64 | 0.028% | 0.037% |
| B4 | 121 | 128 | 0.560% | 0.711% |

每个编码结果都核对了特殊 Token、右侧 Padding 和 attention mask。MacBERT 审计发现删除空格会改变 1,753 条训练样本和 491 条验证样本的 Token ID，其中 1,723 和 482 条含 ASCII 字母或数字。中文用单个空格连接冻结文本的原有词元，保留代码切换与数字的边界。

## 执行

```powershell
.venv/Scripts/python.exe -m nlp_text_classification.cli batch-b-s1 data-audit
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-b-s1 smoke
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-b-s1 run
.venv/Scripts/python.exe -m nlp_text_classification.cli batch-b-s1 status
.venv/Scripts/python.exe -m nlp_text_classification.cli batch-b-s1 report
```

`artifacts/batch-b-s1/progress.md` 为实时快照，`progress.jsonl` 追加记录槽位、epoch、更新计数、验证和预测进度。Run 的完整状态每轮原子保存。使用同一协议重新执行 `run` 可以复用已验证结果，恢复中断的槽位。CUDA OOM 触发一次微批调整，报告保留原失败尝试。正式 S1 只通过训练与验证入口运行。
