# Batch B S4 正式训练 Runbook

三个 Seed 已完成训练和验证冻结。下文的训练命令是已执行协议的记录。正式测试与产物复核见本页末尾。 [验证冻结报告](../../reports/batch-b-s4/formal/report-s4.md)

正式入口以 `a55c761` 的 `A_winner_foreach_evalfast` 为基线，使用现有 WSL2 Ubuntu 22.04 的隔离环境 `~/.cache/emotionlab-s4-perf/venv`。入口在离线模式下校验模型 Revision、官方文件指纹、加速包、训练数据和 Tokenizer 指纹。正式 Run 只读取 GoEmotions 冻结的 train 和 validation 分区，测试留待独立冻结流程。 [S4 推荐配置](../../reports/batch-b-s4/performance/recommended-training.json)

## 训练协议

| 项目 | 配置 |
| --- | --- |
| 模型 | `Qwen/Qwen3.5-4B-Base`，Revision `1001bb4d826a52d1f399e183466143f4da7b741b`，NF4 double quant，BF16 compute |
| 可训练参数 | 248 个线性投影，LoRA Rank 32、Alpha 64、Dropout 0.05，加官方 28 维 `score` 头，共 65,001,472 个参数 |
| 训练批次 | Micro-batch 16 × 累积 1，动态右填充到 8 的倍数，最大长度 64，同一 Epoch 内随机排列；最后 11 条做一次更新 |
| CUDA 与优化器 | RTX 4070 Ti SUPER 16 GB，FLA、causal-conv1d，`partial_16` 非重入梯度检查点，AdamW `foreach=True`；LoRA LR `1e-4`，分类头 LR `5e-4`，梯度裁剪 1.0 |
| 显存管理 | 启用 CUDA 可扩展分段。最长输入超过 48 token 的训练批次使用 32 层非重入重计算，批次前后释放闲置缓存，返回常规批次时恢复 `partial_16`。全量验证前也释放闲置缓存。工程验收线为 2,048 MiB，正式运行预警线为 2,560 MiB。 |
| 更新预算 | `ceil(43211 / 16) = 2701` 步/轮，`2701 × 5 = 13505` 步；Linear Warmup 为 `ceil(13505 × 0.06) = 811` 步 |
| 学习率计数 | 初始化 Scheduler step 0，两组 LR 均为 0；每次 AdamW 更新后调用一次 Scheduler，尾批同样计一步；step 811 达到基准 LR，step 13,505 衰减到 0 |
| 验证 | 每个完成的 Epoch 使用全量 5,416 条，稳定长度排序，batch 32，评分后按原样本顺序回填；多标签 BCE 与 Sigmoid；校验 `0.05–0.95` 的 19 个阈值 |
| 选择 | Validation Macro-F1 最佳轮次；阈值并列时先取距 0.5 最近者，再取较小阈值；轮次 F1 在 `1e-12` 内并列保留较早轮次；连续 2 轮未改善早停，上限 5 轮 |
| 随机性 | 每个 Seed 独立初始化 `score` 和 LoRA；每轮样本排列由该 Seed 和零基 Epoch 共同确定，Checkpoint 保存 RNG、排列和下一批样本指纹 |

## 启动和监控

人工审阅通过后，从仓库根目录在 PowerShell 中分三晚串行执行。每条命令的 `--preflight` 指向已经验收的预检目录，入口校验预检签收与当前运行绑定。每晚等上一个进程退出后再执行下一条。

```powershell
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_formal.sh train --seed 42 --preflight prelaunch-v6
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_formal.sh train --seed 123 --preflight prelaunch-v6
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_formal.sh train --seed 2026 --preflight prelaunch-v6
```

为保留 WSL、Python 和 CUDA 的完整控制台输出，可把当晚的启动命令通过 `2>&1 | Tee-Object -FilePath artifacts/batch-b-s4/formal/seed-42-terminal.log -Append` 保存，并在其他两晚替换文件名中的 seed。结束时以训练退出码和 `COMPLETE` 事件共同确认成功。

在独立 PowerShell 窗口查看进度和监督日志，`status` 可在任意终端单次查询。

```powershell
Get-Content artifacts/batch-b-s4/formal/runs/seed-42/progress.jsonl -Tail 12 -Wait
Get-Content artifacts/batch-b-s4/formal/runs/seed-42/watchdog.jsonl -Tail 12 -Wait
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_formal.sh status --seed 42
```

### 实时监控面板

只读面板不接触训练进程、不创建 CUDA 上下文，也不写入 Run 目录：

```powershell
.venv\Scripts\python.exe apps\monitor\s4_monitor.py --seed 42 --open
```

面板默认监听 `127.0.0.1:8765`，展示总进度、轮次、loss、梯度范数、学习率、吞吐、显存、每轮 Macro-F1、每轮耗时、有效 token 比例、逐标签指标、事件流和看门狗状态。`--digest` 输出一次文本摘要；服务每 300 秒把 `digest.json` 和 `digest.txt` 写入 `artifacts/batch-b-s4/formal/monitor/seed-<seed>/`。切换 Seed 时替换 `--seed`，省略时自动跟随最新的 `runs/seed-*`。

`progress.md` 是可实时查看的状态快照。`progress.jsonl` 追加每 25 步的更新、已用学习率、训练 Loss、验证行数、Checkpoint、退出原因和资源异常。`watchdog.jsonl` 记录子进程 PID、终止原因、资源释放和重试决定。`status` 读取最后完整 Checkpoint 的已持久化状态，进程活跃时用 `progress.jsonl` 看实时位置。`EPOCH` 事件包含验证阈值、Macro-F1、历史最佳轮次和本轮验证耗时。

## 产物、验收和恢复

每个 Seed 使用 `artifacts/batch-b-s4/formal/runs/seed-<seed>/`。`run.json` 保存协议与指纹，`checkpoints/step-XXXXXXXX/` 保存不可变的检查点。启动时保存 step 0，此后每 500 次更新及每轮验证结束保存一次。每个检查点包含 `checkpoint.json`、`resume.pt` 和 `adapter/`，验证检查点另含 `predictions_validation.jsonl`、`metrics_validation.json`、`thresholds.json`。`checkpoint.json` 的文件 SHA-256、AdamW 两组学习率及 497 个可训练张量的状态用于恢复核验。发布可使用最佳检查点的 `adapter/` 与官方基座 Revision，`resume.pt` 用于续训。按 `state.best.global_step` 定位最佳检查点。

| 中断原因 | 处理 |
| --- | --- |
| Ctrl+C、进程退出、宿主重启 | 保留日志及完整的 `step-*`。用同一 Seed、分支和预检签收执行 `resume`，从最后完整提交点及其样本游标重放。最坏需要重做约 500 次更新，若中断发生在验证中还会重做该次验证。 |
| 初始化在 step 0 保存前失败 | 启动条件恢复后重试 `train`，入口核对 `run.json` 并用同一 Seed 从头初始化。 |
| 整卡余量短时低于 2,560 MiB | 正式训练记录 `VRAM_WARNING` 并继续，余量恢复到 2,816 MiB 后重新允许下一次预警。工程预检另核验全程余量。 |
| CUDA OOM、整卡余量持续至少 15 秒低于 1,024 MiB，或进度超时 | 监督进程处理本次子进程。启动、资源审计和模型加载 300 秒无新进度为超时，验证 75 秒，其他阶段 120 秒。终止先发 SIGTERM，12 秒未退出则发 SIGKILL。确认进程退出且 45 秒内 `nvidia-smi` 读到至少 10,240 MiB 空闲，才自动重试一次。有完整检查点时从该点恢复。 |
| 显存释放未确认、重试再失败、主机内存门禁 | 查看 `watchdog.jsonl` 和终端日志，确认本次 GPU Worker 状态，排查宿主 GPU 占用和 WSL 内存。条件恢复后执行 `resume`，尚在 step 0 前则使用 `train`。更改批次、精度或内核需要独立协议和预检。 |
| Checkpoint、模型、训练数据或环境指纹不符 | 保存 `HALTED` 记录，查明差异并找回匹配版本。`.pending-*` 目录为中断时的暂存目录，恢复时只扫描已提交的 `step-*`。 |

```powershell
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_formal.sh resume --seed 42 --preflight prelaunch-v6
```

启动前签收涵盖 `16×64` 完整训练更新、真尾批、AdamW 驻留时两次全量验证、文件指纹、跨进程 RNG 和下一真实批次重放。当前入口签收为 `prelaunch-v6`；v6 仅向 `TRAIN` 事件追加 `gradient_norm`、`valid_tokens` 和 `padded_tokens`，训练数值与 v5 逐字节一致（除 `adapter_config.json` 的 `target_modules` 集合顺序）。运行中逐步检查 Loss、梯度范数和显存。每轮完成后校验导出的规范样本 ID、28 维目标、原始分数、阈值和预测，并逐个复算阈值网格的 Macro-F1。每个 Seed 的完成条件为 `COMPLETE`、可定位的最佳检查点、逐轮完整验证证据和可审计的早停状态。三个 Seed 完成后再进行独立选模冻结审计。 [正式入口验收](../../reports/batch-b-s4/formal/report.md)

## 时间和存储预算

S4 短测在匹配批次下测得 21.885 samples/s，完整验证约 41.36 秒。原预算为 5 轮每 Seed 2.81 小时，保守 3.23 小时。正式入口最多保留 33 个检查点，预检一次含验证归档的保存耗时约 11.9 秒，按该耗时估算保存约增加 7 分钟。每晚先留 3.5–4 小时，拿到首个正式 Epoch 的时间后再更新余下两晚预算。给每个 Seed 预留约 25 GiB，三个 Seed 合计约 75 GiB，另加验收证据、现有模型缓存和运行余量。 [S4 预算证据](../../reports/batch-b-s4/performance/time-budget.json)

需重做工程验收时使用新 `--attempt`，先运行有限更新，再启动独立进程验证。两个命令都只在 `formal/preflight/<attempt>/` 写入。

```powershell
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_formal.sh preflight --attempt prelaunch-recheck
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_formal.sh verify-preflight --attempt prelaunch-recheck
```

## 冻结后的独立测试与复核

独立入口绑定 `97ce29261f3844e715731f62f7859df2397db2e7381744bc8a0db86cff4e5d15`。`audit` 复算验证证据及核对分区清单。`test` 校验 WSL2 与模型文件后读取一次源测试分区，将校验过的副本留在本地版本目录，三个 Seed 共用此副本。`replay` 读取副本和已发布预测，重新核对各项指标，无需 CUDA 推理。成功的 Seed 目录禁止覆盖。有故障时用原命令只补齐缺失 Seed。

```powershell
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_test.sh audit
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_test.sh test
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_test.sh replay
```

测试本地版本目录为 `artifacts/batch-b-s4/formal/test/<freeze_sha256>/`。`progress.md`、`progress.jsonl` 记录阶段和进度，`seed-*/` 保存逐样本分数、指标及文件指纹。`contract.json`、`pretest-audit.json`、`tokenization.json` 和 `acceptance.json` 记录选择边界、输入及复算验收。原逐样本证据和缓存保留在本地。 [正式测试报告](../../reports/batch-b-s4/formal/final-test.md)
