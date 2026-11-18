# Batch B S4: Qwen3.5-4B 分类训练性能专项调查总结报告 (`report.md`)

**调查日期:** 2026-11-18  
**基线基准:** 冻结提交 `124c1a2`（分支 `s3-baseline-124c1a2`）及 `reports/batch-b-s3/`  
**S4 工作分支:** `batch-b-s4` | **正式三种子训练启动状态:** `false` | **冻结测试集访问:** `false`  
**实际主动 GPU 短测累计耗时:** `1,965.812 s`（**32.76 分钟**，远低于 1 小时目标上限与 4 小时预算红线）

---

## 1. 执行摘要（Executive Summary）

S3 WSL2 路径（`micro_batch=1, accum=16`, 32 层全开 non-reentrant GC, `eval_batch=1`）单次更新耗时 `13.81 s`（`1.159 samples/s`），导致 GoEmotions `3 seeds × 5 epochs` 预计耗时 **`160.13 小时`（约 6.7 天）**。本轮 S4 在严格保持 Batch B 同科学口径（同模型 Revision `1001bb4d`、NF4 + BF16 计算、248 个 LoRA r=32/α=64 + `score` 分类头、有效 batch=16、BCE、相同训练样本顺序与学习率调度）的前提下，完成源码审计、CUPTI Trace 剖析与分阶段对照实验：

1. **真实根因查明：** S3 并非算力饱和，也不是静态 pad-to-64（H3 已证伪）。`micro_batch=1` 将单次更新切分为 **`272,574` 个微内核调用**，引发 **`~72%` 的纯 CPU 发射空隙**（无插桩 wall `14.56 s` 中 GPU 合并忙时仅 **`4.087 s`**），同时将 GPU 功耗压低至 `48.6 W`、SM 时钟压至 `1,467 MHz`、16× 放大 NF4 反量化流量（`23,808` 次 `kDequantizeBlockwise`），并使 32 层全开 Checkpoint 重算额外消耗 **`32.8%`** 时间；此外 `eval_batch=1` 拖慢验证集至 `4.79 samples/s`（15 次约 `4.71 h`）。
2. **A 类无损加速突破：** 优胜配置 **`A_winner_foreach_evalfast`**（`micro_batch=16, accum=1`, `dynamic_mult8` 不排序, `gc_mode="partial_16"`, `AdamW(foreach=True)`, `eval_batch=32 sorted`）实现：
   - **训练吞吐：** **`21.8852 samples/s`**（相对 S3 `1.1586` 提升 **`18.889×`**，单更新 `0.7311 s`）；
   - **验证吞吐：** **`130.9508 samples/s`**（相对 S4 `eval_b1` 提升 `28.7×`，15 轮验证由 `4.71–4.95 h` 压缩至 **`0.1723 h ≈ 10.3 min`**）；
   - **端到端时间预算：** `3 seeds × 5 epochs` 中性总预算由 **`160.13 h` 骤降至 `8.430 h`**（乐观 `8.010 h`，保守 `+15%` **`9.690 h`**）；`1 seed × 5 epochs` 为 **`2.810 h`**；
   - **显存与可靠性验收：** 整卡最小剩余显存 **`4,167 MiB`**（> `2,048 MiB` 门禁 2 倍余量），248 个 LoRA + `score` 梯度全部非零，跨进程恢复（`PID 548 -> PID 747`）验证最大分数误差 **`0.0`**、RNG 与优化器步长逐位一致。
3. **最终建议：** **`GO_FULL`**（等待人工批准后使用 `recommended-training.json` 启动三种子实验；本轮未自行启动正式训练）。

---

## 2. 口径严格分离说明

| 数据来源口径 | 本报告中标注方式 | 用途范围 |
|---|---|---|
| **S3 冻结实测** | `S3 WSL2` / `comparison.json`（`1.1586 train / 4.793 val s/s`） | 历史基线对照与倍率分母 |
| **S4 本机实测** | 所有 `S4_0` / `B*` / `P*` / `GC_*` / `A_winner` / `eval_sweep` / `B1` / `B2` / CUPTI Trace 数据 | 本轮核心性能与数值证据 |
| **外部资料参考** | Unsloth 官方文档 Qwen3.5 指南、HF Transformers/PyTorch 文档链接 | H8/架构机理辅证（不作本机速度承诺） |
| **公式情景推算** | `time-budget.json`（`0.95× / 1.00× / 1.15×`，含开销不假设早停） | `1×5`、`3×2`、`3×3`、`3×5` 耗时预算 |

---

## 3. 核心交付物总表

| 项目 | 数值 / 状态 | 说明与证据来源 |
|---|---|---|
| **最优 S4 方案（A 类）** | **`A_winner_foreach_evalfast`** | `micro=16, accum=1, dynamic_mult8 (unsorted), gc_mode=partial_16, NF4+FP32 stream, AdamW(foreach=True), eval_b32_sorted` |
| **相对 S3 端到端加速** | **`18.889×`**（训练） / **`27.3×`**（验证） / **`19.0×`**（3×5 总耗时 `160.13h -> 8.43h`） | [`acceptance.json`](./acceptance.json), [`time-budget.json`](./time-budget.json) |
| **训练吞吐（train）** | **`21.8852 samples/s`**（有效 token **`367.60 tok/s`**，均值 **`0.7311 s/upd`**） | [`experiments.csv`](./experiments.csv) |
| **验证吞吐（validation）** | **`130.9508 samples/s`**（全量 5,416 行实测 `41.36 s`） | [`validation-matrix.csv`](./validation-matrix.csv) |
| **峰值整卡显存 / 最小剩余** | **峰值 `11,895 MiB` / 最小剩余 `4,167 MiB`**（PyTorch peak alloc `8,617.8 MiB`） | `A_winner_foreach_evalfast/result.json` |
| **精度与恢复验收** | **PASS**（post-13 vs S3 prob_mean=`0.00215`, flips=`5/1344` 均在边缘；跨 PID 恢复 error=`0.0`） | [`accuracy-and-recovery.md`](./accuracy-and-recovery.md) |
| **单种子 5 Epoch 预算** | **2.810 h** 中性（乐观 `2.670 h` / 保守 `3.230 h`） | [`time-budget.json`](./time-budget.json) |
| **三种子 3 / 5 Epoch 预算** | 3 Epoch: **5.062 h**（`4.81–5.82 h`）；5 Epoch: **8.430 h**（`8.01–9.69 h`） | [`time-budget.json`](./time-budget.json) |
| **额外工程复杂度** | **极低**（仅调 batch/eval/GC 层开关与 `foreach=True`，不改数学定义与第三方内核） | [`recommended-training.json`](./recommended-training.json) |
| **置信等级** | **高**（全 5,416 验证全测、独立 PID 恢复 `0.0` 误差、连续 8 稳态窗口低波动 `p95=0.817 s`） | [`acceptance.json`](./acceptance.json) |

### 被证伪的假设（Falsified Hypotheses）
本轮共有三项关键既有直觉/假设被实测证伪：**第一，`H3`「输入总是静态填充到 64 浪费大量算力」被证伪**——源码与 `padding-shapes.csv` 证明 S3 已使用动态 8 倍数填充，`micro=1` 在 43,211 条训练集上有效 token 占比达 `82.90%`（而非固定 64 的 `25.98%`），且在 `micro=8` 下开启更新内按长度排序（`P5_mult8_sorted`: `7.17 samples/s`）因动态 shape 频繁切换及频率降低反而比不排序（`B4_micro8`: `9.81 samples/s`）慢 27%。**第二，`H1` 副推论「`nvidia-smi` 利用率 `36.1%` 仅代表低占空比 Kernel 计算」被部分修正**——CUPTI 证明真正的 `gpu_all_active` 在 `14.56 s` 更新内只有 `4.087 s`，高达 `~72%` 是核间纯 CPU 空隙，GPU 大部分时间根本没有执行内核。**第三，`B1`（去除 FP32 残差流强转）与 `H6`（BF16 LoRA 16GB 下全面更优）在目标约束下被部分证伪**——`B1` 在 `micro=16` 虽省出 `2.2 GiB` 显存但吞吐无提升（`21.33 vs 21.89 s/s`）且数值翻转增至 3 倍（`15/1344`）；`B2` BF16 LoRA 在 `micro=8, GC=off` 下跌破 2GiB 门禁（`min_free=250 MiB`）且与 S3 NF4 基线初始概率最大差达 `0.367`（`68/1344` 翻转）。此外 `H7`（`torch.compile`）因 Qwen3.5 存在 `8` 处原生 Graph Break（`16` 子图）且 WSL 独立环境缺失 `g++` 未能在不改编译链下运行。

---

## 4. 性能瓶颈按耗时贡献排序

详见 [`bottleneck.md`](./bottleneck.md) 与 [`timeline-analysis.md`](./timeline-analysis.md)：
1. **#1 `micro_batch=1, accum=16` 微内核碎片化（最致命，独立解决即提速 `14.01×`）：** 每更新 `272,574` 个内核造成 `~72%` CPU 发射空隙（其中 `ChunkGatedDeltaRule` 仅 GPU `183 ms` 却消耗 CPU `3,065 ms`），同时 `16×` 重复调用 `11,904` 次 NF4 反量化并落入低占空比（`4.9–11.2%`）`16×16` WMMA GEMM。
2. **#2 全 32 层非重入 Gradient Checkpointing（次大开销，再提速 `+26.7%`）：** Update 5 事件跨度 `4.358 s / 13.272 s`（`32.8%`），迫使前向与 NF4 反量化额外重算一整遍。改为间隔 16 层检查点（`partial_16`）在 `micro=16` 下将速度从 `16.23` 推升至 `20.57 s/s`（配合 `foreach=True` 独立进程达 **`21.89 s/s`**），同时保住 `4,167 MiB` 安全显存。
3. **#3 验证集 `eval_batch=1` 与次要系统开销：** 验证 `b=1 -> b=32 + length_sort` 将 15 轮评估耗时从 `4.71–4.95 h` 压缩至 **`0.172 h`**（提速 `28.7×`）；`AdamW(foreach=True)` 与无碎片隔离运行进一步稳定至 `21.89 s/s`；而 `elementwise_cast`（`64,880` 次 FP32<->BF16 转换）在 `micro=16` 下不再是主瓶颈。

---

## 5. 假设检验总表（H1–H8）

| ID | 假设内容 | 实测结论 | 关键证据与处置 |
|---|---|---|---|
| **H1** | `micro-batch=1` 导致小 GEMM 与低效率 | **成立（超预期）** | `micro 1->2->4->8->16`: `1.099 -> 2.545 -> 4.468 -> 9.813 -> 16.233 s/s`（`14.01×` S3），`min_free=6,057 MiB`。**采纳 `micro=16`**。 |
| **H2** | Checkpoint 重算占比可削减 | **成立（半层最优）** | 占 Update 5 耗时 `32.8%`。`micro=16` 下 `partial_16` 提速至 `20.57 s/s`（独立 +foreach `21.89 s/s`，`4,167 MiB` free）；`attn8`（`842 MiB`）与 `off`（`31 MiB`）均跌破 `2,048 MiB` 门禁。**采纳 `partial_16`**。 |
| **H3** | 实际右 Padding 总是到 64 浪费算力 | **证伪** | 源码已使用动态 8 倍数（`micro=1` 有效率 `82.9%`）。组内按长度排序在 `micro=8` 下因 shape 切换反而减速（`7.17 vs 9.81 s/s`）。**训练关闭排序，仅验证启用有序 batch=32**。 |
| **H4** | 验证 `batch=1` 过慢可大幅放批 | **成立** | 全量 5,416 行验证 `b=32 sorted` 达 `130.95 s/s`（`28.7×` vs `b=1`），15 轮总计 `0.1723 h`，`min_free=6,082 MiB`，与 `b=1` 仅 `4/1,344` 边缘翻转。**采纳 `eval_batch=32 sorted`**。 |
| **H5** | CPU 提交 / Kernel Launch / NF4 反量化制约 GPU | **成立** | Trace 证实 GPU 忙时仅 `4.087 s`（`~72%` 空隙），NF4 反量化 `31.9%` GPU 时，`16×16` WMMA 占空比 `4.9–11.2%`，`ChunkGatedDeltaRule` CPU `3,065 ms` vs GPU `183 ms`。**已在 `micro=16` + `partial_16` 根本缓解**。 |
| **H6** | BF16 LoRA 在 16GB 下可运行且更快（B 类） | **部分证伪 / 不推荐当前使用** | `B2 micro=1 GC=full` 虽可运行（`1.438 s/s`, `3,626 MiB` free）但 `micro=8 GC=off` 跌破门禁（`min_free=250 MiB`）；且 NF4 vs BF16 第 0 步即产生 `0.3666` 最大概率偏差（`68/1,344` 翻转），破坏 S3 NF4 可比口径。 |
| **H7** | `torch.compile` / CUDA Graph 提升短序列吞吐 | **在当前环境失败 / 受阻** | `dynamo.explain` 确认存在 **`8` 处 Graph Break（16 子图）**（`chunk_gated_delta_rule` `@compiler.disable`、`causal_conv1d` 非连续 `out=`、`fla` `get_device_properties` 等），且隔离 WSL 无 `g++` 触发 `InvalidCxxCompiler`。**禁用 `torch.compile`**。 |
| **H8** | Unsloth / 其他框架可更快且等价 | **架构/口径核验未通过无损替换条件** | 外部官方核验（`docs.unsloth.ai`）：Unsloth 原生面向 `AutoModelForCausalLM` 因果生成与定制 Mamba/DeltaNet Triton 内核，对 HF `AutoModelForSequenceClassification` 需外部封装；且其官方明确不建议对 Qwen3.5 使用 4-bit QLoRA（*"higher than normal quantization differences"*，推荐 BF16 LoRA ~10GB），与 S3 NF4 口径不直接等价。**仅列外部审计，不作为无损替换**。 |

---

## 6. 全量实验矩阵与失败 / 未测项记录

### 6.1 Category A（保持同数学定义）全表

| 实验 ID | Micro/Accum | Padding/Sort | GC | Foreach | Eval | 吞吐 (s/s) | 倍率 vs S3 | Update 均值 (s) | Min Free (MiB) | 状态 |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| `S4_0_baseline` | 1 / 16 | mult8 / unsort | full | False | b1 | 1.0987 | 0.948× | 14.562 | 5,202 | PASS |
| `B2_micro2` | 2 / 8 | mult8 / unsort | full | False | b1 | 2.5448 | 2.196× | 6.287 | 5,836 | PASS |
| `B3_micro4` | 4 / 4 | mult8 / unsort | full | False | b1 | 4.4680 | 3.856× | 3.581 | 5,843 | PASS |
| `B4_micro8` | 8 / 2 | mult8 / unsort | full | False | b1 | 9.8127 | 8.469× | 1.631 | 5,828 | PASS |
| `B5_micro16` | 16 / 1 | mult8 / unsort | full | False | b1 | 16.2329 | 14.010× | 0.986 | 6,057 | PASS |
| `P0_static64_micro1` | 1 / 16 | static64 / unsort | full | False | b1 | 0.9830 | 0.848× | 16.277 | 5,444 | PASS |
| `P2_static64_micro8` | 8 / 2 | static64 / unsort | full | False | b1 | 9.0812 | 7.838× | 1.762 | 5,057 | PASS |
| `P4_mult8_sorted_micro4` | 4 / 4 | mult8 / sort | full | False | b1 | 4.4342 | 3.827× | 3.608 | 5,711 | PASS |
| `P5_mult8_sorted_micro8` | 8 / 2 | mult8 / sort | full | False | b1 | 7.1706 | 6.189× | 2.231 | 5,452 | PASS |
| `P6_mult1_sorted_micro8` | 8 / 2 | mult1 / sort | full | False | b1 | 8.1242 | 7.012× | 1.969 | 5,419 | PASS |
| `GC_micro1_partial16` | 1 / 16 | mult8 / unsort | partial_16 | False | b1 | 1.3569 | 1.171× | 11.792 | 5,388 | PASS |
| `GC_micro1_off` | 1 / 16 | mult8 / unsort | off | False | b1 | 1.9343 | 1.669× | 8.272 | 5,274 | PASS |
| `GC_micro8_partial16` | 8 / 2 | mult8 / unsort | partial_16 | False | b1 | 11.5004 | 9.926× | 1.391 | 4,366 | PASS |
| `GC_micro8_attn8` | 8 / 2 | mult8 / unsort | partial_attn8 | False | b1 | 13.2158 | 11.406× | 1.211 | 3,417 | PASS |
| `GC_micro8_off` | 8 / 2 | mult8 / unsort | off | False | b1 | 13.9335 | 12.026× | 1.148 | **858** | **FAIL_VRAM_HEADROOM** |
| `GC_micro16_partial16` | 16 / 1 | mult8 / unsort | partial_16 | False | b1 | 20.5726 | 17.756× | 0.778 | 3,094 | PASS |
| `GC_micro16_attn8` | 16 / 1 | mult8 / unsort | partial_attn8 | False | b1 | 23.5709 | 20.343× | 0.679 | **842** | **FAIL_VRAM_HEADROOM** |
| `GC_micro16_off` | 16 / 1 | mult8 / unsort | off | False | b1 | 26.0423 | 22.476× | 0.614 | **31** | **FAIL_VRAM_HEADROOM** |
| **`A_winner_foreach_evalfast`** | **16 / 1** | **mult8 / unsort** | **partial_16** | **True** | **b32_sort** | **21.8852** | **18.889×** | **0.731** | **4,167** | **PASS (Winner + Recovery PASS)** |

### 6.2 Category B（数值/精度口径变更）与 `torch.compile` 失败记录

- **`B1_nf4_bf16_stream`**（`nf4_bf16_stream`, `micro=16, partial_16`）：`PASS` 显存安全（`6,399 MiB` free，较 `A_winner` 多释放 `2,232 MiB`），但吞吐 **`21.3321 s/s`（未超越 A 类优胜且慢于 21.89 s/s）**、冷启动第一步 `32.91 s`、13 步后相对 S3 翻转升至 **`15/1,344`**（`prob_max=0.06275`），**不推荐采用**。
- **`B2_bf16_lora_micro1_gcfull`**：`PASS`（`1.4381 s/s`，`min_free=3,626 MiB`），证明同配置 `micro=1` 去 NF4 反量化虽快于 S4-0（`1.44 vs 1.10 s/s`），但总耗时仍过慢且与 NF4 S3 基线存在 `0.3666` 最大初始概率差（`68/1,344` 翻转）。
- **`B2_bf16_lora_micro8_gcoff`**：**`FAIL_VRAM_HEADROOM`**（`15.0417 s/s`，`window_min_free=1,120 MiB`，worst-case `[8, 64]` 下 **`250 MiB` free < 2,048 MiB**，仍慢于 A 类 `21.89 s/s`）。
- **`compile_submodules_default` & `compile_full_default_dynamic`**：均记为 **`FAIL_COMPILE_ERROR`**（`InductorError: InvalidCxxCompiler: g++ missing`；同时注意 Stage C 紧接 `GC_micro16_off` 16,031 MiB 近满显存后同进程运行的 `_measure_uncompiled_ref` 因 WDDM 内存压力退化为 `135.52 s`，已隔离标为受污染读数，干净稳态对照应以 `GC_micro16_partial16` `0.7777 s` 为准，该异常亦进一步证明必须严守 `>= 2,048 MiB` WDDM 显存安全边界）。

### 6.3 Category C（改变科学问题/方法）提案与尚未验证的优化项（诚实边界）

按任务书规定，Category C 仅作理论提案、不改动本轮正式口径：
1. **全局按长度分桶批采样（Global Length-Sorted / Bucketed Batches，偏 A/B 边界）：** `padding-shapes.csv` 理论核算表明，若打破严格原始随机打乱顺序（改用全局分桶采样），`micro=16` 有效 token 比可由 `48.2%` 回升至 **`82.9%`**（减少 ~42% 计算 token）；因 `A_winner` 已达 `8.43 h` 且不希望引入分桶改变训练样本统计分布，本轮**未运行**，列为备选。
2. **仅对后 N 层（或 8 层 FullAttention）微调 LoRA：** 将 trainable adapter 从 248 缩减约 `2×–4×`，可进一步释放显存并降低反向梯度成本，但改变了全网络可适应表示，需单独消融对照。
3. **冻结 Qwen3.5-4B 主干缓存最后 Token 隐状态 + 轻量分类器（Linear/MLP Probe）：** 仅需 1 次前向特征提取（以 `130.95 s/s` 全套 43k+5.4k 数据特征缓存不到 **7 分钟**！），后续千次超参扫描仅秒级完成；但退化为固定特征线性可分性问题，无法替代端到端 QLoRA 微调。
4. **换用 Qwen3.5-0.8B / 2B 或蒸馏学生模型：** 若探索模型尺寸 scaling law 可大幅提速，但改变科学对照问题本身。
5. **本轮其他尚未实测组合：** (a) `B1_nf4_bf16_stream` 搭配 `GC_off`（借助其省出的 `2.2 GiB` 显存探索是否稳于 2GiB 以上且超过 `21.89 s/s`）；(b) `B2_bf16_lora` 搭配 `micro=16, partial_16/full`（作为独立 B 类高精度 BF16 实验）；(c) 在 WSL2 安装 `build-essential (g++)` 后对消除了 8 个 Graph Break 的代码评估 `torch.compile`；(d) Unsloth 外挂自定义 `score` 头方案的实机吞吐测试。

---

## 7. 直接回答任务书第 8 节 10 个核心问题

1. **13.81 秒 / update 的前三大真实 GPU/CPU 瓶颈是什么？证据在哪里？**  
   ① **`micro_batch=1, accum=16` 造成的微内核碎片与 `~72%` CPU 发射空隙**（CUPTI 见 `272,574` 个内核、GPU 仅活跃 `4.087 s`、功耗 `48.6 W`、`ChunkGatedDeltaRule` CPU 耗时 `3,065 ms` 是 GPU `183 ms` 的 `16.7×`、`23,808` 次 NF4 反量化占 GPU `31.9%`、`16×16` WMMA 占空比 `4.9–11.2%`；增批至 16 即跳涨 `14.01×`）；  
   ② **32 层全开 non-reentrant Gradient Checkpointing 重算**（Update 5 事件耗时 `4.358 s` 占 `32.8%`，在 `micro=16` 下改为 `partial_16` 再提速 `+26.7%`）；  
   ③ **`eval_batch=1` 验证开销与次级 FP32 残差转换流**（`eval b1 -> b32 sorted` 将验证提速 `28.7×`；Trace 测得 `64,880` 次 FP32<->BF16 cast）。证据见 `profile-summary.json`、`operator-top10.csv` 与 `experiments.csv`。

2. **GPU 36% 利用率主要是 Kernel 间空隙还是 Kernel 内计算效率？是否有可核实的 SM / Tensor Core 证据？**  
   **两者共存，但主因是高达 ~72% 的 Kernel 间纯 CPU 发射空隙，同时叠加热内低占空比小矩阵。** `14.56 s` 内所有内核与拷贝并集活跃仅 `4.087 s`；WMMA `16×16` GEMM 在 CUPTI 中估算 `est. achieved occupancy` 仅 `4.94%–11.17%`。由于 WSL2 虚拟化环境不开放裸机 `nsys/ncu` SM/Tensor Core 硬件指令计数器，该底层硬件计数器诚实标为不可得，未予编造；而当消除空隙并增大矩阵块至 `micro=16` 后，实测板卡功耗从 `48.6 W -> 150.6 W`、SM 时钟从 `1,467 -> 2,775 MHz`，直接验证算力释放。

3. **实际输入 Tensor 是否一直 Padding 到 64？有效 Token 与计算 Token 比例是多少？**  
   **不是。** 源码 `DevicePartition.batch(chunk, multiple=8)` 已是动态 8 倍数填充。在 `micro=1` 下，训练集（43,211 条，平均有效长度 `16.628`）平均填充长度仅 `20.059`，**有效 Token 比为 `82.90%`（`718,506 / 866,752`）**，远高于静态 pad-to-64（`25.98%`）。仅当不排序扩大 `micro` 时，有效率才会下降（`micro=16` 为 `48.24%`）。

4. **micro-batch 1→2→4→8（及 16），哪个速度最高、哪个最稳定、哪组 OOM？为什么？**  
   保持 `effective=16, GC=full` 下：**全组（1/2/4/8/16）全部通过显存门禁，零 OOM**；吞吐单调飙升：**`1.099 -> 2.545 -> 4.468 -> 9.813 -> 16.233 samples/s`**。其中 **`micro=16`** 速度最高、且最小剩余显存反而最大（`6,057 MiB` free > `5,202 MiB` @ `micro=1`），原因是一步清零减少累加缓冲区存活波动并将 16 倍碎片化发射降为 1 倍；若追求极端显存稳定冗余，`micro=8/16 (GC=full)` 均拥有 `5.8–6.0 GiB` 空闲，而配合 `partial_16` 的 **`micro=16 (partial_16 + foreach)` 兼得最高速度 `21.8852 samples/s` 与 `4,167 MiB` 稳定余量**（`p95 update = 0.817 s`）。

5. **Eval batch 增大后，15 次 validation 的总预算从约 4.7h 降到多少？**  
   从 S3 的约 `4.71 h`（S4 `eval_b1` 下约 `4.95 h`）骤降至 **`0.1723 小时`（约 `10.34 分钟`）**——基于全量 5,416 行实测 `eval_b32_sorted_full5416` 的 `130.951 samples/s`（单轮 `41.36 s`，最小剩余显存 `6,082 MiB`）。

6. **Checkpoint 重算最多能实际净节省多少，是否需要多占过高显存？**  
   在最优 `micro=16` 下：彻底关闭（`GC=off`）最高可至 `26.04 samples/s`（净省 `+60.4%` vs `16.23`），但剩余显存跌至 **`31 MiB`**（严重违背 `>= 2,048 MiB` 门禁）；`partial_attn8`（`23.57 s/s`）剩余显存仅 **`842 MiB`** 同样不安全；**仅开启 16 个交替层检查点（`partial_16`）是最优平衡点**：稳态推升至 **`20.57 -> 21.89 samples/s`（相对 `full` 再净快 `+26.7%~+34.8%`）**，而整卡依然稳留 **`3,094–4,167 MiB`** 空闲显存。

7. **BF16 LoRA、`torch.compile`、Unsloth 在这一精确任务上是否真的净加速？兼容性、数值、成本如何？**  
   - **BF16 LoRA (`B2`)**：同条件下（`micro=1, GC=full`）略快于 NF4（`1.438 vs 1.099 s/s`）但依然太慢，而关闭 GC 到 `micro=8` 会触发 `FAIL_VRAM_HEADROOM`（`250 MiB` free < `2,048 MiB`）；更重要的是它与 S3 NF4 在第 0 步就有 `0.3666` 概率偏差（`68/1,344` 翻转），属于改变底座精度的 B 类路线，目前性价比与一致性均不如 `A_winner`；  
   - **`torch.compile`**：在本任务混合架构中碰到 **`8` 处 Graph Break（16 子图）**（`chunk_gated_delta_rule` 被显式 `@torch.compiler.disable`、`causal_conv1d` 非连续 `out=` 等），且隔离 WSL2 环境缺 `g++` 报错退出，当前无正收益且不可用；  
   - **Unsloth**：原生面向 `CausalLM` 文本生成与自研 Triton 内核，对 `SequenceClassification + score` 需额外适配包装；官方还明确劝阻对 Qwen3.5 用 4-bit QLoRA（存在较大定量误差，建议 BF16 LoRA），无法无损等价替代当前 S3/S4 任务口径。

8. **最优、可重现方案的 train/validation 吞吐和 24h/36h 预算分别是多少？保守区间是什么？**  
   - 方案：**`A_winner_foreach_evalfast`**
   - 实测吞吐：**Train = `21.8852 samples/s`**，**Validation = `130.9508 samples/s`**，跨 PID 恢复 **`0.0` 误差**。
   - 总时间预算（不假设早停触发）：  
     - **`1 seed × 5 epochs`**: 中性 **`2.810 h`**，保守 **`3.230 h`**（24h 阈值余量超过 7 倍）；  
     - **`3 seeds × 3 epochs`**: 中性 **`5.062 h`**，保守 **`5.818 h`**；  
     - **`3 seeds × 5 epochs`**: 中性 **`8.430 h`**，保守 **`9.690 h`**（均远低于 24h / 36h 决策线！）。

9. **如果目标依旧达不到，是因为 4B 规模的不可避免前后向成本，还是当前实现的特殊问题？支持证据是什么？**  
   **实测已大幅超越 24h/36h 目标（8.43h 即可完成完整 3 seeds × 5 epochs）。** 事后证据明确证明，S3 原先 `160 小时` 的瓶颈几乎全部来自**可修复的运行时碎片与保守配置（`micro=1`、`GC=full`、`eval=1`）**，而非 4B 规模本身的硬性极限——仅靠不改数学的 A 类系统配置修复便释放了 **`18.89×`** 的真实算力。

10. **请提出 `GO_FULL / GO_PILOT / DEFER / DROP_4B` 四者之一作为建议，说明另外三个选项为什么次优，并明确正式训练尚未启动，等待人工审阅。**  
    - **建议：`GO_FULL`**（采用 `A_winner_foreach_evalfast` 配置 [`recommended-training.json`](./recommended-training.json) 运行完整 3 seeds × 5 epochs，全量耗时 `8.01–9.69 h` 可一夜完成）。  
    - **其余三项次优理由：** ① **`GO_PILOT` 次优**，因完整三轮已压进 10 小时内（单轮 2.8h），顺序跑完三轮即天然先产出 seed 42 且总成本可控，没必要因耗时顾虑人为割裂多种子；② **`DEFER` 次优**，因当前 `21.89 s/s` 已超 `2.7 s/s` 暂缓线 `8.1×`，且 `4,167 MiB` 显存与跨进程恢复均验收通过，暂无理由搁置；③ **`DROP_4B` 次优**，因 4B 不可运行的算力假设已被打破，放弃会损失同系列规模对比价值（C 类小模型或线性探针宜作后续并列实验而非替代）。  
    - **状态重申：正式 3 seeds × 5 Epoch 训练尚未启动（`formal_training_started: false`），冻结测试集零触碰，停止在人工审阅入口等待决策。**

---

## 8. 交付文件与证据索引

- 调查报告：[`report.md`](./report.md)（本文件）、[`s4-baseline-audit.md`](./s4-baseline-audit.md)、[`bottleneck.md`](./bottleneck.md)、[`timeline-analysis.md`](./timeline-analysis.md)、[`accuracy-and-recovery.md`](./accuracy-and-recovery.md)、[`decision-memo.md`](./decision-memo.md)
- 结构化数据：[`baseline.json`](./baseline.json)、[`environment.txt`](./environment.txt)、[`padding-shapes.csv`](./padding-shapes.csv)、[`operator-top10.csv`](./operator-top10.csv)、[`batch-matrix.csv`](./batch-matrix.csv)、[`padding-matrix.csv`](./padding-matrix.csv)、[`validation-matrix.csv`](./validation-matrix.csv)、[`checkpoint-matrix.csv`](./checkpoint-matrix.csv)、[`compile-matrix.csv`](./compile-matrix.csv)、[`experiments.csv`](./experiments.csv)、[`experiments.json`](./experiments.json)
- 预算与验收：[`time-budget.json`](./time-budget.json)、[`recommended-training.json`](./recommended-training.json)、[`acceptance.json`](./acceptance.json)
- 剖析与恢复原始证据（本地 `artifacts/`，未随报告进 Git）：`artifacts/batch-b-s4/performance/traces/s4_1_baseline_update6.json`（`sha256: a2f8146d...`）、`artifacts/batch-b-s4/performance/profile-summary.json`、`artifacts/batch-b-s4/performance/runs/A_winner_foreach_evalfast/recovery.json`
