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

## CUDA 预检验收

2026-11-18，RTX 4070 Ti SUPER 上的四组 BF16 小样本训练、反传、最优权重重载和验证分数独立复算通过。实际资源配置如下。

| 模型 | 微批次 | 梯度累积 | 有效批次 | 峰值 allocated GiB | 峰值 reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 32 | 1 | 32 | 1.29 | 1.42 |
| B2 | 16 | 2 | 32 | 3.14 | 3.42 |
| B3 | 16 | 2 | 32 | 7.87 | 8.60 |
| B4 | 16 | 2 | 32 | 2.57 | 2.71 |

B3 首先从微批 8 和梯度检查点开始。Windows 内存映射重载触发访问异常，恢复优化器时也记录了分配失败。修复后以流式权重加载逐张量恢复 CUDA 状态，继续保留 6 GiB 系统内存暂停线。随后通过微批 16、关闭梯度检查点的吞吐预检，有效批次仍为 32。完整状态恢复探针复核模型、AdamW、scheduler 和 RNG，保存的恢复状态、逐轮历史、阈值与验证预测逐字节一致。

正式门禁为 `artifacts/batch-b-s1/preflight-runs/f82a3adc9179/gate.json`，SHA-256 为 `932c5646caa4cfc230912350574b17d779b555ec9fe7eabfeadce9c5dd2f342a`。恢复探针见 `artifacts/batch-b-s1/b3-resume-probe/audit.json`。首次故障与最终验收的终端日志分别保留于 `smoke-terminal.log`、`smoke-verified-terminal.log` 和 `smoke-final-terminal.log`。29 项 pytest 与 Ruff 通过。

## 正式 S1 交付验收

2026-11-18，四个基础槽位和四个候选槽位均通过。入选 Run 的验证 Macro-F1 为 B1 0.4958、B2 0.5305、B3 0.5285、B4 0.9076。验证审计逐 Run 复核原始预测、阈值候选、最佳轮次、早停、输出头及运行设置。四张对照曲线、Markdown 与 JSON 从已有产物重新生成并进行指纹核对。

训练引用代码提交 `6ea3f2c`。原始队列与审计位于 `artifacts/batch-b-s1/formal/queue.json` 和 `formal/audit.json`，交付复核为 `artifacts/batch-b-s1/delivery-audit.json`。Batch A 冻结清单、47 份正式测试、原报告与两轮补充报告均通过指纹核对。 [S1 报告与曲线](../../reports/batch-b-s1/report.md) · [S2 评估入口](../plans/batch-b/s2-followups.md)
