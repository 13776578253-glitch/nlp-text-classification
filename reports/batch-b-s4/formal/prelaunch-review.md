# Batch B S4 预启动复核报告

- 时间：2026-11-19 11:30（Asia/Shanghai）
- 分支：`codex/batch-b-s4-formal`
- 基线提交：`5165ba4`（父提交 `a55c761`）
- 审查人：主 agent
- 独立复核：`independent-review.md`（证据项 **PASS**；原 B1 已撤回）
- 修订：Rev 5 — 撤回 B1：正式 `protect()` 在 WSL 内读 `/proc/meminfo`，不是 Windows `resource_status()`
- **结论：PASS**。代码、数据、运行时绑定和证据全部通过；WSL 门禁 **22.4 GiB available ≥ 8 GiB**；Windows 主机 9.14 GiB 是 headroom 提示，不是正式入口门禁。训练尚未启动。

## 1. 本轮改动

| 文件 | 改动 |
| --- | --- |
| `src/.../experiments/batch_b_s4_formal.py` | 仅在 `TRAIN` 事件追加 `gradient_norm`、`valid_tokens`、`padded_tokens`；训练数值、批次、精度、优化器、阈值、显存策略均未变 |
| `apps/monitor/s4_monitor.py` | 新增只读监控后端；不 import torch、不建 CUDA 上下文；只写 `artifacts/batch-b-s4/formal/monitor/` |
| `apps/monitor/index.html` | 英文浅色监控页；8 图 + 4 表；事件只保留生命周期/预警/恢复类 |
| `tests/unit/test_s4_monitor.py` | 9 项监控单测；增量 JSONL、状态机、ETA、快照、digest、HALTED 时效 |
| `tests/unit/test_qwen_s4_formal.py` | 测试替身补齐 `gradient_norm` / token 字段 |
| `reports/batch-b-s4/formal/*` | v6 签收、reference、checkpoint 清单；v5 证据保留为 `prelaunch-v5-*` |
| `docs/runbooks/batch-b-s4-formal.md` | 三条 train 与 resume 全部改为 `--preflight prelaunch-v6` |

未改动：`recommended-training.json`（哈希钉死）、`batch_b_s3.yaml`、`s3-wsl-runtime.lock.txt`、`model-resources.json`、冻结数据、其余 12 个绑定源码。

## 2. 硬绑定核对

### 2.1 13 个绑定源码

`reports/batch-b-s4/formal/acceptance.json` 的 `binding.source_sha256` 与实际文件逐项一致，**13/13 PASS**：

```text
batch_b_s4_formal.py            690e417d1936...f28e79  PASS
batch_b_s4_formal_state.py      a45fc4e7f0de...8702f   PASS
batch_b_s4_formal_preflight.py  18415021f678...275d1   PASS
batch_b_s4_formal_watchdog.py   61d57d3eeb8d...1f524   PASS
batch_b_s4_runner.py            46ce75fcb016...72f96   PASS
batch_b_s4_telemetry.py         b87b71dd365e...7792    PASS
batch_b_s3.py                   9afb23c4eed0...93e98   PASS
qwen_s3.py                      a0d32d9d16e2...0e4ba   PASS
qwen_s3_streaming.py            472ac7333c96...cddf    PASS
transformer_runner.py           f093c60633b4...0849    PASS
qwen_s3_inputs.py               aecef5bb7c21...0e34    PASS
batch_metrics.py                258ce9db3cc4...af88    PASS
scripts/batch_b_s4_wsl_formal.sh 7aa109b6885d...67a3   PASS
```

与 v5 相比，只有 `batch_b_s4_formal.py` 的哈希变化（`eb8ae85e…` → `690e417d…`）；其余 12 个绑定源码与 v5 完全一致。

### 2.2 钉死配置

| 对象 | 结果 |
| --- | --- |
| `reports/batch-b-s4/performance/recommended-training.json` | `7d43d2d62bf9…be66`，与代码 `CANDIDATE_SHA256` 一致 |
| `configs/experiment_suites/batch_b_s3.yaml` | 与 recipe `training_suite_sha256` 一致 |
| `configs/batch_b/s3-wsl-runtime.lock.txt` | 与 recipe `package_lock_sha256` 一致 |
| `reports/batch-b-s3/performance/model-resources.json` | 与 recipe `model_resources_sha256` 一致 |
| `data/processed/goemotions/v1/training.json` | `fbd69186f3c8…c98c`，与 recipe `training_manifest_sha256` 一致 |
| `data/processed/goemotions/v1/dataset_manifest.json` | `f9f503dc3da8…725d`，与数据集配置一致 |

`recommended-training.json` 的 `status` 仍是 `awaiting_review_s4_candidate`。**不要编辑该文件**：它的 SHA-256 被 `CANDIDATE_SHA256` 钉死，改动会使正式入口拒绝启动。

### 2.3 v6 验收

- `status: PASS`，`formal_training_started: false`。
- `reports/.../acceptance.json` 与 `artifacts/.../prelaunch-v6/acceptance.json` 完全一致。
- `reports/.../reference.json` 的 SHA 等于 acceptance 的 `reference_sha256`；`reports/.../checkpoint-manifest.json` 的 SHA 等于 `checkpoint_sha256`。
- 4 次提交更新 + 11 条真实尾批完成；第 5 次更新由 reference 记录并由独立进程重放；AdamW/Scheduler 第 4 步，497 个参数状态驻留。
- 全量验证 5,416 行：49.253 s；独立重载验证 38.769 s；检查点保存 11.492 s。
- 151,648 项验证分数逐项一致，`max_abs_score_error: 0.0`；RNG 探针、样本游标、下一批第 5 次更新精确一致。
- 整卡最低余量 2,821 MiB，高于 2,048 MiB 工程门禁；低于 v5 的 3,948 MiB。

### 2.4 v5 / v6 等价性

- Checkpoint `files` 只差 `adapter/adapter_config.json`；该文件仅 `target_modules` 的集合序列化顺序不同（12 个目标模块集合相同）。
- `parameters`、`updates`、`state`、`cursor` 完全一致。
- `adapter_model.safetensors`、`resume.pt`、`predictions_validation.jsonl`、`metrics_validation.json`、`thresholds.json` 逐字节一致。
- `16x64` 压测的 loss 1.146449、梯度范数 14.015074、两组学习率、梯度明细、`gradient_checkpointing` 一致。
- 尾批 loss 1.043926、梯度范数 11.413060；验证指标与 `next_update` 一致。
- 训练 payload 与检查点的训练数值（loss、梯度、学习率、梯度明细、尾批、验证指标、`next_update`）逐项一致；唯一的训练 payload 差异是 `adapter_config.json` 的 `target_modules` 集合顺序。运行时读数另有差异：验证 43.68→49.25 s、重载 54.19→38.77 s、保存 11.11→11.49 s、samples/s 124.0→110.0、进程号 1991→1989、压测 1.2499→1.2114 s、重载显存 6,538→5,355 MiB、整卡最低余量 3,948→2,821 MiB。这些来自桌面负载/调度，不是训练策略变化。

## 3. 数据与 tokenizer

- `training.json` 的 split 指纹：`train.jsonl` `31b27182…5eab`、43,211 行；`validation.jsonl` `017da8aa…7ce7`、5,416 行；实际文件哈希一致。
- `dataset_manifest.json` `f9f503dc…725d`，与数据集配置一致。
- 重新执行了正式入口的 `_binding(load_recipe())` 与 `checked_inputs(recipe, progress)`：
  - `BINDING_EQUAL True`（运行时、源码、计划、候选哈希全部匹配 v6 acceptance）。
  - `INPUTS_OK: 43,211 train / 5,416 validation`；`effective_batch_size=16`、`updates_per_epoch=2701`、`total_updates=13505`、`warmup_updates=811`。
  - 编码形状：train `(43211, 28)`，validation `(5416, 28)`；tokenizer/encoded 哈希匹配 recipe。
- 测试集未读取；正式入口只加载 train + validation。

## 4. 运行时

- WSL2 Ubuntu 22.04，cwd `/mnt/d/Desktop/code/nlp-text-classification`；隔离 venv 存在。
- Python 3.12.14；torch 2.9.1+cu128；transformers 5.10.2；peft 0.19.0；bitsandbytes 0.49.2；accelerate 1.16.0。
- FLA 0.5.1；flash-linear-attention 0.5.1；causal-conv1d 1.7.0；triton 3.5.1。
- GPU：RTX 4070 Ti SUPER，capability [8, 9]；BF16 与 FLA 快路径已由 v6 验收覆盖。
- 当前整卡空闲 13,839 MiB（会波动）；v6 验收最低余量 2,821 MiB；2,560 MiB 预警线，1,024 MiB 持续 15 s 强杀，10,240 MiB 才允许自动重试。
- 无 `runs/` 目录；无正式训练 worker；当前只有监控进程与无关桌面进程。
- D 盘空闲 2,647.8 GiB；三个 Seed 预计 75 GiB，余量充足。
- 主机 commit 余量 43.96 GiB，充足。
- `.batch-a-gpu.lock` 文件存在，但 v6 预检与 watchdog 故障夹具已验证获取与释放；当前无 worker。
- 注：主机内存、GPU 空闲、磁盘、commit、无 worker 等为审查时的 **live** 读数，未归档为文件。Windows 主机的 `resource_status()` 为 9.14 GiB，但这不是正式入口的 `protect()` 门禁；后者在 WSL 内读 `/proc/meminfo`。

### 4.1 内存门禁澄清（原 B1 撤回）

```text
Ubuntu WSL free -m:
  total 24,034 MB  used 1,267 MB  cache 9,261 MB  available 22,420 MB
  swap 16,384 MB free
Docker WSL distro:
  total 24,611 MB  free 13,781 MB  cached 8,181 MB  available 22,939 MB
Windows host resource_status(): 9.14 GiB free, 45.06 GiB commit
vmmemWSL: 7,162 MB working set
```

- `training/resources.py` 在 WSL 内 `sys.platform == "linux"`，`memory_available()` 读 `/proc/meminfo`，**不是** Windows 的 `resource_status()`。正式入口的 `protect()` 看到的是 Ubuntu 的 22.4 GiB available，8 GiB 门禁通过；v6 preflight 的 `RESOURCES` 21.76 / 37.76 也来自同一 WSL 分支。
- Ubuntu 内 `used` 仅 1.27 GiB，`buff/cache` 9.26 GiB；无 Python、torch、训练进程（仅 systemd/snapd 等系统进程）。`vmmemWSL` 7.1 GB 是 WSL2 的工作集/页缓存，不是失控进程。
- 已执行 `wsl --terminate Ubuntu-22.04`；Ubuntu 现为 Stopped。主机空闲仍为 9.14 GiB，说明 7.1 GB 主要来自 **Docker Desktop 的 `docker-desktop` WSL distro**（cached 8.18 GB），而不是 Ubuntu。
- Docker Desktop 进程仍在运行，`com.docker.build` 处于活动状态。**训练门禁已通过，不依赖 Docker**；若需要释放 Windows 主机内存，可退出 Docker Desktop 或执行 `wsl --shutdown`（训练启动会重启 Ubuntu）。为避免打断可能的镜像构建，未直接终止 Docker。

## 5. 监控

- 服务运行在 `http://127.0.0.1:8765/`；审查时页头（live）：**Idle · seed 42 · uptime 0s · waiting for events · VRAM 13,839 MiB**。
- `digest.txt`：`S4 monitor | Idle | no training run found`。
- 只读：不 import torch、不建 CUDA 上下文、不写 Run 目录；只写 `artifacts/batch-b-s4/formal/monitor/seed-42/`。
- 训练启动后会自动读取 `runs/seed-42/progress.jsonl`、`watchdog.jsonl`、检查点与 `nvidia-smi`；不会创建 `runs/`。
- 事件表只显示生命周期、VRAM 预警、RESUMED/HALTED/COMPLETE 等；TRAIN/VALIDATE/EPOCH/例行 CHECKPOINT 由图表和轮次表展示。
- 监控不替代 watchdog；显存/进度回收仍由正式入口的独立监督进程执行。

## 6. Runbook

`docs/runbooks/batch-b-s4-formal.md` 已更新：

```powershell
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_formal.sh train --seed 42 --preflight prelaunch-v6
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_formal.sh train --seed 123 --preflight prelaunch-v6
wsl.exe -d Ubuntu-22.04 --exec bash scripts/batch_b_s4_wsl_formal.sh train --seed 2026 --preflight prelaunch-v6
```

恢复命令同样指向 `prelaunch-v6`。每夜预留 3.5–4 小时；先跑 seed 42，拿到首轮实测时间后再更新余下两晚预算。

## 7. 风险清单

| 编号 | 风险 | 处置 |
| --- | --- | --- |
| B1 | 主机物理内存 7.36 GiB < 8 GiB | **阻断启动**；释放内存至 ≥12 GiB，重跑 `resource_status()` |
| R1 | GPU 余量 2,821 MiB，仅高于 2,560 预警线 261 MiB | 关闭 Wallpaper Engine/浏览器硬件加速；监控 VRAM_WARNING；低于 1,024 MiB 15 s 会被 watchdog 终止 |
| R2 | 任何绑定文件被编辑 | v6 签收失效；必须重新 `preflight` + `verify-preflight`，训练命令改用新 attempt |
| R3 | 编辑 `recommended-training.json` | 其哈希被 `CANDIDATE_SHA256` 钉死；不要改 `status` 字段 |
| R4 | 改批次/精度/内核 | 必须新协议与新预检；不能热修补 |
| R5 | 内存门禁失败留下部分 Run | 删除 `runs/seed-42` 后重试；监控已改为仅当前 HALTED 才告警 |
| R6 | 跨后端数值差 | 训练、恢复、正式复算必须锁定同一 WSL2 后端 |
| R7 | 测试集一次性 | 只有三 Seed 完成、独立选模冻结后才读取 test |
| R8 | Git 工作区有意未提交 | 训练身份由源码哈希与数据指纹确定；不要重命名/移动绑定文件 |
| R9 | `PYTORCH_CUDA_ALLOC_CONF` 弃用警告 | 仅警告；绑定仍校验该变量，保持现状 |
| R10 | `adapter_config.json` 的 target 顺序差异 | 集合相同，非数值差异；已在报告中记录 |

## 8. 复核清单

| 项目 | 结果 |
| --- | --- |
| 13 个绑定源码哈希 | PASS |
| candidate / suite / package lock / model resources / training manifest / dataset manifest | PASS |
| v6 acceptance PASS、`formal_training_started=false` | PASS |
| reference / checkpoint 哈希与归档副本 | PASS |
| 全程恢复 151,648 分数、最大误差 0、RNG/游标/下一步 | PASS |
| 训练 payload 等价（运行时读数差异见 §2.4） | PASS |
| 数据 43,211 / 5,416、编码哈希与形状 | PASS |
| 正式 `_binding` 重跑 `BINDING_EQUAL True` | PASS |
| 无 `runs/`、无正式 worker | PASS |
| 监控 Idle 页头、只读、digest | PASS |
| Runbook 三条 train + resume 指向 v6 | PASS |
| pytest 68 项、Ruff、`git diff --check` | PASS（`report.md` 已同步为 68） |
| **WSL VM MemAvailable ≥8 GiB（正式 `protect` 门禁）** | **PASS（22.4 GiB；v6 `RESOURCES` 21.76/37.76）** |
| Windows 主机空闲（headroom，非门禁） | 9.14 GiB；`vmmemWSL` 7.1 GB 主要为 Docker 缓存 |
| GPU 最低余量 ≥2,560 MiB | PASS（v6 2,821，余量偏小） |
| 磁盘 ≥75 GiB | PASS（2,647.8 GiB） |

## 9. 独立复核

独立只读 agent 的证据复核：v5/v6 只有 `batch_b_s4_formal.py` 哈希与 `adapter_config.json` 的 target 顺序差异，训练数值、runbook、monitor 写域、68 项测试声明均一致；4 处表述问题已在 Rev 2/3 修正。其原 **CONDITIONAL PASS** 的唯一条件是 B1；该条件基于 Windows `resource_status()`，而正式 `protect()` 在 WSL 内读 `/proc/meminfo`（22.4 GiB available；v6 `RESOURCES` 21.76/37.76），因此 **B1 撤回，修正结论为 PASS**。

## 10. 证据路径

- `reports/batch-b-s4/formal/acceptance.json`（v6 签收）
- `reports/batch-b-s4/formal/reference.json`（v6 首进程证据）
- `reports/batch-b-s4/formal/checkpoint-manifest.json`（v6 检查点清单）
- `reports/batch-b-s4/formal/prelaunch-v5-acceptance.json`（v5 历史签收）
- `reports/batch-b-s4/formal/report.md`（验收报告）
- `artifacts/batch-b-s4/formal/preflight/prelaunch-v6/`（原始进度、终稿、检查点）
- `reports/batch-b-s4/performance/recommended-training.json`（候选配置，勿改）
- `docs/runbooks/batch-b-s4-formal.md`（启动/恢复/监控）
- `apps/monitor/`（只读监控）

复现要点：13 个源码与 6 个钉死配置用 SHA-256 对照 acceptance；正式 `_binding` 在 WSL venv 中重跑并与 acceptance 的 `binding` 比较；`checked_inputs` 重新执行 tokenizer/encoded 校验；主机内存用 `nlp_text_classification.training.resources.resource_status()` 检查。
