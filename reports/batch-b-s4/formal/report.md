# Batch B S4 正式训练入口验收

基线为 `a55c761` 的 `A_winner_foreach_evalfast`。验收使用 RTX 4070 Ti SUPER、WSL2 S4 隔离环境及同一官方 Qwen3.5-4B-Base Revision。正式入口锁定 43,211 条训练数据、5,416 条验证数据、28 维标签顺序及 Tokenizer 指纹。验收只使用 train、validation 和合成输入，三个正式 Run 等待人工审阅。 [正式训练 Runbook](../../../docs/runbooks/batch-b-s4-formal.md) · [S4 推荐配置](../performance/recommended-training.json)

## 试验记录与修复

| 验收尝试 | 观测与处理 |
| --- | --- |
| `prelaunch-v1` | `16×64` 压力更新后整卡最小余量 1,807 MiB，门禁停止。 |
| `prelaunch-v2` | 在长批次和验证前释放闲置 CUDA 缓存。压力更新后的最低余量为 1,969 MiB，PyTorch 峰值分配 10,540 MiB、峰值保留 10,946 MiB，门禁停止。 |
| `prelaunch-v3` | 启用 CUDA 可扩展分段，压力更新完成。尾批阶段检查到全程最小余量 2,013 MiB，门禁停止。 |
| `prelaunch-v4` | 保留上述分配器设置。输入长于 48 token 时启用 32 层非重入重计算，并在批次前后释放闲置缓存。普通批次沿用 `partial_16`。独立恢复验收通过，整卡最小余量 4,206 MiB。 [v4 签收](prelaunch-v4-acceptance.json) |
| `prelaunch-v5` | 保留 v4 的训练和显存分配策略，加入正式运行软预警、独立监督、进程组回收和一次检查点重试。入口验收通过。 [v5 正式签收](prelaunch-v5-acceptance.json) |
| `prelaunch-v6` | 仅向 `TRAIN` 事件追加已计算的 `gradient_norm`、`valid_tokens`、`padded_tokens`，训练与显存策略不变；重新预检与独立恢复验收通过。 [v6 正式签收](acceptance.json) |

S4 原始全量 padding 统计中，56 和 64 宽度合计 19/2,701 个批次。正式训练的每轮排列随 Seed 和 Epoch 变化。正式预算沿用原训练吞吐估算，并考虑长批次与 500 步检查点，每夜预留 3.5–4 小时。 [原始 padding 统计](../performance/padding-shapes.csv) · [时间预算](../performance/time-budget.json)

## `prelaunch-v4` 有限步 CUDA 验收

- `16×64` 合成输入的 1,024 个 token 全部有效；由正式代码完成一次前向、反向和 AdamW 更新。Loss 为 1.146449，裁剪前的梯度范数为 14.015074，248 个 LoRA A、248 个 LoRA B 和 1 个分类头张量均有有限非零梯度。该阶段 PyTorch 峰值分配为 5,788.8 MiB，峰值保留为 8,578 MiB。
- 11 条真实尾批调用一次 AdamW 和一次 Scheduler。保存时二者均为第 4 步，497 个参数的 AdamW 状态已驻留。Adapter、分类头和优化器状态的指纹均相对初始化发生变化。
- 全量验证用 `eval_batch=32` 稳定长度排序，再按原始样本顺序输出 `5,416×28` 项分数。首遍验证为 42.96 秒。19 个候选阈值的 Macro-F1 由 sklearn 独立复算，最优阈值为 0.05，Macro-F1 为 0.073559。这是第 4 步的工程验收分数。
- 新的 CPU 验收进程核对了 12 个源码指纹、8 个 Checkpoint 文件指纹、5,416 个唯一验证样本及 19 个候选阈值。正式 Run 目录尚未创建。

## 正式运行监督验收

正式运行短时低于 2,560 MiB 时记录预警并继续。低于 1,024 MiB 持续至少 15 秒或进度超时时，独立的 CPU 监督进程终止本次训练进程组。进程退出且 45 秒内确认空闲显存至少 10,240 MiB 后，才允许自动重试一次。有已提交检查点时恢复其模型、优化器、Scheduler、RNG 和样本游标。

WSL2 的三个故障夹具使用真实进程组和文件锁。无进度夹具和持续临界显存夹具均终止了忽略 SIGTERM 的进程，并验证锁已释放。第三个夹具在终止后启动新的进程，加载合成 step 500 并退出成功。显存阈值分支使用固定注入的读数。 [监督进程结构化证据](watchdog-evidence.json)

在 `prelaunch-v5` 的首次完整验证中，5,416 条样本耗时 43.68 秒。v4 与 v5 的 Adapter（含分类头）、逐样本验证输出、指标和阈值文件逐字节一致。 [v5 首进程证据](prelaunch-v5-reference.json) · [v5 检查点清单](prelaunch-v5-checkpoint-manifest.json)

v5 在进程 `362` 保存 AdamW 和 Scheduler 第 4 步，进程 `1991` 独立重载后用 54.19 秒完成全量验证。151,648 项分数的最大绝对误差为 0，阈值网格一致。Python、NumPy、CPU 和 CUDA 的 RNG 探针及下一批样本游标精确恢复，重放后的第 5 步更新和参数指纹一致。两次进程的最小整卡显存余量为 3,948 MiB。 [v5 正式签收](prelaunch-v5-acceptance.json)

## `prelaunch-v6` 遥测修订

v6 只修改 `batch_b_s4_formal.py` 的 `TRAIN` 事件，追加 `gradient_norm`、`valid_tokens` 和 `padded_tokens` 三个已经由 `run_update` 计算并返回的字段。批次、精度、优化器、阈值、显存策略和检查点格式均未改变。`acceptance.json` 的 `source_sha256` 中，只有 `batch_b_s4_formal.py` 从 `eb8ae85e…` 变为 `690e417d…`，其余 12 个绑定文件与 v5 完全一致。重新执行 4 次提交更新 + 第 5 次重放、11 条真实尾批、全量验证和跨进程恢复后，验收为 `PASS`。 [v6 正式签收](acceptance.json) · [v6 首进程证据](reference.json) · [v6 检查点清单](checkpoint-manifest.json)

- 4 次提交更新和真实尾批完成；第 5 次更新在独立进程重放；AdamW 与 Scheduler 停在第 4 步，497 个参数状态驻留。全量验证 5,416 行耗时 49.25 秒，检查点保存 11.49 秒；独立进程重载后的验证耗时 38.77 秒。
- 151,648 项验证分数逐项一致，最大绝对误差为 0；RNG 探针、样本游标、下一批样本指纹和第 5 次更新一致。
- 整卡最低余量为 2,821 MiB，高于 2,048 MiB 工程门禁，但低于 v5 的 3,948 MiB。v5→v6 的余量下降与验收时更高的桌面 GPU 占用一致（具体占用比例未归档；按总显存、峰值保留与最低余量推算约 4.3 GiB）。正式训练前建议关闭 Wallpaper Engine、浏览器硬件加速等非必要 GPU 占用，以恢复 v5 的余量水平。
- 与 v5 逐字节对比：Adapter 权重、`resume.pt`、`predictions_validation.jsonl`、`metrics_validation.json` 和 `thresholds.json` 完全一致；`adapter_config.json` 仅 `target_modules` 的集合序列化顺序不同，12 个目标模块集合相同。`16×64` 压测的 loss、梯度范数、学习率和梯度明细逐项一致。
- 新增的只读监控 `apps/monitor/` 通过 `progress.jsonl`、`watchdog.jsonl`、`nvidia-smi` 和轮末检查点展示进度、loss、梯度、显存、Macro-F1、ETA 和监督事件；不写入 Run 目录，也不在 13 个绑定源码内。

## 单元测试与证据路径

单元测试初次运行把整除情况下的“尾批余数”误写成 16，修正为 0。初次 Ruff 检查发现未使用的测试变量和预检入口导入顺序，分别补上验证统计断言与整理导入。最终 68 项 pytest、Ruff 和 Git diff 空白检查通过。新增测试覆盖真实预算、11 条尾批、三 Seed 排列、恢复后从样本游标继续、验证顺序、阈值网格、最佳轮次、早停、长批次重计算切换、低显存预警和有限重试；新增的 `tests/unit/test_s4_monitor.py` 覆盖增量 JSONL 读取、状态机、ETA、快照聚合与 digest。

尝试日志保存在 `artifacts/batch-b-s4/formal/preflight/<attempt>/`。每次尝试有 `preflight-terminal.log` 和增量 `progress.jsonl`，跨进程恢复使用同目录的 `verification-terminal.log`。S3、S4 性能证据及原冻结实验目录与 `a55c761` 的 Git diff 为空。
