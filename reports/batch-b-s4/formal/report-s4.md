# Batch B S4 正式训练、冻结与测试交接报告

本页记录验证冻结时的状态。“测试集未读取”适用于本页所述训练与冻结阶段。后续评估见 [独立正式测试报告](final-test.md)。

- 日期：2026-11-19
- 分支：`codex/batch-b-s4-formal`
- 冻结清单：`reports/batch-b-s4/formal/frozen/97ce29261f3844e715731f62f7859df2397db2e7381744bc8a0db86cff4e5d15.json`
- 冻结 SHA-256：`97ce29261f3844e715731f62f7859df2397db2e7381744bc8a0db86cff4e5d15`
- **测试集状态：未读取（`test_read: false`）**

本报告覆盖三件事：三个 Seed 的正式训练结果、验证证据的冻结、以及交给后续独立流程一次性读取测试集的交接契约。

## 1. 结论摘要

Qwen3.5-4B-Base + NF4 QLoRA 在 GoEmotions 28 标签任务上，三个 Seed 的验证 Macro-F1 为 **0.5421 ± 0.0024**（均值 ± 样本标准差，ddof=1）。

| Seed | 最佳轮次 | Step | 验证 Macro-F1 | 阈值 | 停止原因 | 实际轮数 |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 2 | 5402 | 0.540334 | 0.20 | early_stopping | 4 |
| 123 | 2 | 5402 | **0.541193** | 0.30 | early_stopping | 4 |
| 2026 | 1 | 2701 | **0.544786** | 0.15 | early_stopping | 3 |
| **均值** | — | — | **0.542104** | — | — | 3.67 |

其余指标（逐 Seed 均值）：

| 指标 | 均值 | 逐 Seed |
| --- | --- | --- |
| Macro-F1 | 0.542104 | 0.540334 / 0.541193 / 0.544786 |
| Micro-F1 | 0.598561 | 0.596787 / 0.606083 / 0.592815 |
| PR-AUC（macro） | 0.558209 | 0.554878 / 0.557925 / 0.561822 |
| Hamming Loss | 0.039763 | 0.041042 / 0.035272 / 0.042975 |

对照 Batch B Encoder 正式测试：现代 BERT-large 三种子测试 Macro-F1 均值 0.5378。Qwen 尚未读测试集，因此**当前只能在验证集口径上作比较**，不能宣称已超越。

## 2. 训练配置（已冻结）

| 项目 | 值 |
| --- | --- |
| 基座 | `Qwen/Qwen3.5-4B-Base`，Revision `1001bb4d826a52d1f399e183466143f4da7b741b` |
| 量化 | NF4 double quant，BF16 计算，`score` 头不量化 |
| 适配 | LoRA rank 32 / alpha 64 / dropout 0.05，248 个目标模块 + 28 维 `score` 头，65,001,472 可训练参数 |
| 批次 | micro-batch 16 × 累积 1，动态右填充到 8 的倍数，最大长度 64 |
| 优化器 | AdamW `foreach=True`；LoRA LR `1e-4`，头 LR `5e-4`；权重衰减 0.01 / 0.0；梯度裁剪 1.0 |
| 调度 | Linear，warmup `ceil(13505 × 0.06) = 811` 步；每轮 2,701 步，上限 5 轮 |
| 显存策略 | 可扩展分段；长于 48 token 的批次启用 32 层全量重计算，常规批次 `partial_16` |
| 早停 | patience 2，上限 5 轮 |
| 验证 | 每轮全量 5,416 条，batch 32，按长度排序后回填原序 |
| 选择 | Validation Macro-F1；19 点 sigmoid 阈值网格 0.05–0.95；并列容差 1e-12 |

候选配置 SHA-256 为 `7d43d2d62bf9a69bc4474198dc189f14da2f74bfda10b22caae9fdd86a19be66`，与预检 `prelaunch-v6` 绑定的 `CANDIDATE_SHA256` 一致。

## 3. 训练轨迹与过拟合边界

```
epoch      seed 42    seed 123   seed 2026
  1        0.5365     0.5337     0.5448 ★
  2        0.5403 ★   0.5412 ★   0.5243
  3        0.5100     0.5187     0.5088
  4        0.5117     0.5180     (early stop)
  5        (early stop) (early stop)
train loss 0.108→0.011  0.109→0.010  0.113→0.039
```

三次运行呈现一致的过拟合边界：训练 loss 单调下降的同时，验证 Macro-F1 在第 2–3 轮后回落，`patience=2` 在 3–4 轮正常早停。seed 2026 收敛更快（第 1 轮即达峰），因此只跑了 3 轮。这是该配置在 GoEmotions 上的稳定行为，不是随机波动或实现缺陷。

## 4. 冻结内容

### 4.1 冻结范围

`scope: validation_only`，`test_read: false`。冻结模块 `experiments/batch_b_s4_freeze.py` 只调用 `load_training_data`，**从不打开 `test.jsonl`**，也不引用 `load_partition`。测试集文件时间戳保持 `2026-11-17 23:22:24`，与冻结前一致。

### 4.2 每 Seed 的冻结检查点

| Seed | 检查点 | `checkpoint.json` SHA-256 | Adapter SHA-256 | 文件数 |
| --- | --- | --- | --- | --- |
| 42 | `step-00005402` | `c5fa5291688a6063…` | `16a7f48099145482…` | 8 |
| 123 | `step-00005402` | `781f29184048db03…` | `97cdc25c86d4c2cd…` | 8 |
| 2026 | `step-00002701` | `b0c096cec1862c26…` | `c67d0d2fed1da6c6…` | 8 |

完整 64 位哈希见冻结清单的 `runs[].checkpoint_sha256` 与 `runs[].adapter_sha256`。

每个检查点包含 `checkpoint.json`、`resume.pt`（520 MB）、`adapter/`（含 `adapter_model.safetensors`、`adapter_config.json`、`text_config.json`、`README.md`）、`predictions_validation.jsonl`、`metrics_validation.json`、`thresholds.json`。清单中逐一记录了它们的 SHA-256，冻结时全部重算比对通过。

### 4.3 冻结时执行的独立复算

对每个 Seed：

1. **文件完整性**：8 个文件 SHA-256 全部匹配 `checkpoint.json` 的 `files` 映射；`optimizer_step` 与 `scheduler_step` 等于 `global_step`；`parameter_states = 497`。
2. **训练身份**：`run.json` 的 `purpose`、`seed`、`source_sha256`（13 个绑定源码）、`training_manifest_sha256` 与实际文件一致；三个 Run 的 runtime、加速包、分配器设置必须相同。
3. **选择合法性**：`COMPLETE` 事件唯一，停止原因为 `early_stopping`；`state.best` 等于该 Run 各轮 Macro-F1 的最大值；最佳检查点步号与 `state.best.global_step` 一致。
4. **验证分区**：5,416 个样本 ID 与冻结 `validation.jsonl` 顺序一致，`targets` 逐项相等，`raw_labels` 与原始样本标签一致。
5. **阈值复算**：用 `threshold_record` 重算 19 点网格，与 `thresholds.json` 逐项相等；再用 sklearn `f1_score` 独立复算每个候选阈值，误差 ≤ 1e-12；选中阈值与 `macro_f1` 等于冻结记录。
6. **指标复算**：用 `measure` 重算 Macro-F1、Micro-F1、逐标签 P/R/F1/support、Hamming、Subset、PR-AUC，与 `metrics_validation.json` **完全相等**。
7. **分数指纹**：记录 float32 分数的 SHA-256（`score_checksum`），供后续测试阶段比对同一检查点的输出。

### 4.4 选择规则（已冻结）

```json
{
  "metric": "validation_macro_f1",
  "tie_atol": 1e-12,
  "threshold_grid": {"default": 0.5, "range_inclusive": [0.05, 0.95], "step": 0.05},
  "threshold_application": "one global sigmoid threshold for all 28 labels; score >= threshold",
  "epoch_rule": "maximum validation macro-F1; ties keep the earlier epoch",
  "early_stopping": {"patience": 2, "max_epochs": 5},
  "aggregation": "per-seed selected validation score; mean and sample std (ddof=1) across the three seeds"
}
```

阈值取整数刻度构造的 0.05–0.95 共 19 个候选，两端包含。三 Seed 的选定阈值分别是 0.20 / 0.30 / 0.15——均在网格内，且由各自验证集独立选出。

### 4.5 哈希守卫

冻结清单记录了自己与 13 个绑定源码的 SHA-256。`verified_freeze()` 会重新核验：候选配置、全部源码、每个检查点、审计文件。**任何一处被修改都会使冻结失效**。这一点已在本次工作中实测：编辑冻结模块后，旧清单 `accbe551…` 被 `ValueError: Freeze source changed` 拒绝，随后重新生成 `97ce2926…`。

## 5. 交接给独立测试流程

### 5.1 边界

后续流程**尚未执行**。测试集只允许被读取一次，规则如下：

1. 先调用 `verified_freeze()` 核对冻结清单与全部指纹，通过后才可打开测试分区。
2. 读取 `dataset_manifest.json` 的 `splits.test`：`test.jsonl`，**5,427 行**，SHA-256 `351ddec3f36c40b5a7391cf191312db900ee1723cc9469123f05a2a37c46dbba`，须与冻结清单的 `test_partition` 一致。
3. 三个 Run 各用冻结的 `adapter/` + 官方基座 Revision `1001bb4d…` 重载，`micro_batch 32`、按长度排序、回填原序，产出 `5,427 × 28` float32 sigmoid 分数。
4. 应用各 Seed **已冻结的阈值**（0.20 / 0.30 / 0.15），不做任何重新调参或重新选阈值。
5. 用 `measure` 计算指标，并另用 sklearn 独立复算 Macro-F1 / Micro-F1 / Hamming / PR-AUC 交叉验证。
6. 成功输出禁止覆盖；故障只在同一冻结版本补齐未完成项。

### 5.2 必须在报告中声明的限制

- 测试集一次性使用，因此这是**单个冻结版本**的结果，没有第二次机会。
- 三 Seed 只提供 3 个样本点，标准差仅作稳定性描述，不构成显著性检验。
- GoEmotions 的 `neutral` 占验证集支持 1,765/5,416（32.6%），Macro-F1 对它的权重是 1/28，需结合逐标签指标解读。
- 单一全局阈值会压低稀有标签的召回；这是契约规定的口径，不是调参失误。

## 6. 复现与证据路径

### 6.1 冻结复验

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -c "from nlp_text_classification.experiments import batch_b_s4_freeze as f; print(f.verified_freeze()[1])"
```

预期输出 `97ce29261f3844e715731f62f7859df2397db2e7381744bc8a0db86cff4e5d15`。任何指纹不符都会抛异常而非静默通过。

### 6.2 证据

| 内容 | 路径 |
| --- | --- |
| 冻结清单 | `reports/batch-b-s4/formal/frozen/97ce2926….json` |
| 冻结引用 | `reports/batch-b-s4/formal/freeze-ref.json` |
| 独立复算审计 | `reports/batch-b-s4/formal/freeze-audit.json` |
| 入口验收（v6） | `reports/batch-b-s4/formal/acceptance.json` |
| 预启动审查 | `reports/batch-b-s4/formal/prelaunch-review.md` |
| 独立复核 | `reports/batch-b-s4/formal/independent-review.md` |
| Seed 42 Run | `artifacts/batch-b-s4/formal/runs/seed-42/` |
| Seed 123 Run | `artifacts/batch-b-s4/formal/runs/seed-123/` |
| Seed 2026 Run | `artifacts/batch-b-s4/formal/runs/seed-2026/` |
| 终端日志 | `artifacts/batch-b-s4/formal/seed-{42,123,2026}-terminal.log` |
| 只读监控 | `apps/monitor/` |

### 6.3 校验状态

```
pytest        75 passed
ruff          All checks passed
git diff      --check clean
local freezes 3/3 PASS
test.jsonl    untouched (2026-11-17 23:22:24)
workers       none
```

## 7. 未执行与待办

- **未读测试集**：正式测试完全留给后续独立流程。
- **未做跨 Seed 显著性检验**：3 个样本点不足以支撑，报告中只给均值与标准差。
- **未与 Batch A/B Encoder 做统计比较**：口径不同（验证 vs 测试），且需要配对重采样。
- **未生成 LaTeX/图表**：等测试完成后随正式报告一并产出。
