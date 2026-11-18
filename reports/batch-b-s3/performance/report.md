# Batch B S3 · Windows 与 WSL2 性能预检

**推荐 WSL2 Ubuntu 22.04 + CUDA 12.8 + FLA 0.5.1 + causal-conv1d 1.7.0。** 在同一块 RTX 4070 Ti SUPER 上，匹配更新窗口实测 **1.159 样本/秒**，为 Windows PyTorch 回退的 **1.60 倍**。WSL2 通过实际 CUDA 前向、反向、参数更新、发布存档和独立进程恢复。按实测吞吐估算，三种子满 5 轮约需 **155.4 小时纯训练**，另约 **4.7 小时验证**。正式训练等待审阅。

[对照数据](comparison.json) · [验收与证据指纹](acceptance.json) · [建议训练配置](recommended-training.json) · [工程入口](../../../docs/runbooks/batch-b-s3-preflight.md)

## 匹配条件与吞吐

两边使用官方 Qwen3.5-4B-Base Revision `1001bb4d826a52d1f399e183466143f4da7b741b`、同一份 NF4 量化方案、248 个投影的 Rank 32 / Alpha 64 LoRA 和原生 `score` 分类头，共 65,001,472 个可训练参数。共用初始 Adapter、分类头及 RNG，使用同一采样顺序、BF16、micro-batch 1、累积 16、相同的 AdamW 两组学习率和调度步。5 次预热后，比较第 6–13 次更新的相同 128 条训练样本。所选长度为 64、右填充对齐 8。该窗口中最长的原始 token 序列为 37，另以 128 个有效 token 做反向压力检查。 [模型资源](model-resources.json) · [权重加载审计](wsl-load-audit.json) · [LoRA 覆盖明细](wsl-lora-audit.json)

| 指标 | Windows PyTorch 回退 | WSL2 FLA + causal-conv1d |
| --- | ---: | ---: |
| 8 次匹配更新总耗时 | 176.47 s | 110.47 s |
| 平均 / 中位单步时间 | 22.06 / 22.29 s | 13.81 / 13.74 s |
| 样本吞吐 | 0.725 样本/s | 1.159 样本/s |
| 三种子 × 五轮训练估算 | 248.2 h | 155.4 h |
| 训练峰值显存，PyTorch allocated / reserved | 5.42 / 6.15 GiB | 5.42 / 6.17 GiB |
| 第 6–7 次更新 GPU 平均利用率 | 36.9%（42 次采样） | 36.1%（25 次采样） |
| 第 6–7 次更新设备显存峰值，含桌面占用 | 10,243 MiB | 11,391 MiB |
| 128 token 压力测试峰值 allocated | 5.34 GiB | 5.34 GiB |

Windows 原始采样在 5 次预热、30 次稳态更新后提前结束。全部稳态均值为 0.672 样本/s，对应 268.1 小时。表中倍率采用事先对齐的 8 次窗口。GPU 遥测使用独立 Windows 补采与对应的 WSL2 第 6–7 次更新。Windows 补采两次的损失、梯度范数和学习率与原始记录逐项一致，吞吐为 0.693 样本/s。设备整体利用率取自宿主机 `nvidia-smi`，进程显存峰值取自 Torch allocator。

训练估算按 43,211 条 train × 5 epoch × 3 seeds ÷ 稳态吞吐计算。WSL2 的恢复进程对 48 条 validation 用时 10.01 秒，线性估算 15 次完整验证约 4.7 小时。WSL2 两项合计约 160 小时，模型加载、首次 Triton 编译、写盘与故障恢复另计。Windows 训练进程峰值工作集 2.25 GiB、峰值提交量 12.32 GiB。WSL2 训练峰值 RSS 2.83 GiB、恢复峰值约 2.99 GiB。WSL2 匹配窗口设备最少可用 4,671 MiB。

## 瓶颈与内核

独立剖析第 5 次更新，两个后端均处理同一组 16 条样本。下表以 CUDA event 测量各操作跨度，反向已拆出其中的 Checkpoint 重计算。事件时长会受到 CPU 提交间隙影响。Windows 独立剖析时段波动明显，该步 77.12 秒，主吞吐计算使用上表的无插桩匹配窗口。

| 第 5 次更新内的阶段 | Windows 回退 | WSL2 |
| --- | ---: | ---: |
| 数据准备 | 0.016 s | 0.010 s |
| Forward | 24.001 s | 4.503 s |
| Backward，扣除重计算 | 30.789 s | 4.542 s |
| Checkpoint 重计算 | 22.203 s | 4.085 s |
| 梯度裁剪 | 0.006 s | 0.003 s |
| AdamW | 0.079 s | 0.023 s |

训练时间主要集中在模型前向与反向，Checkpoint 重计算占两个剖析步骤约三成。数据准备和 AdamW 改动对当前预算的影响很小。WSL2 运行发现完整 fast path，首个 DeltaNet 层中的 `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule` 与 `causal_conv1d.causal_conv1d_interface.causal_conv1d_fn` 在同一累积更新中各调用 32 次，其中包含前向和重计算，各有 16 次输出收到反向梯度。 [Transformers v5.10.2 Qwen 实现](https://github.com/huggingface/transformers/blob/v5.10.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py) · [FLA v0.5.1](https://github.com/fla-org/flash-linear-attention/tree/v0.5.1) · [causal-conv1d v1.7.0 官方 Linux wheel](https://github.com/Dao-AILab/causal-conv1d/releases/expanded_assets/v1.7.0)

原生 Windows 的 Triton 3.5 与 PyTorch 2.9 有公开兼容路径，FLA 具备单独验证的条件。此次选用的官方 causal-conv1d 发布资产提供匹配的 Linux wheel。原生 Windows 的组合仅完成资料核查，现有环境保持原样。WSL2 使用宿主驱动 610.88 的 CUDA 映射，在现有 Ubuntu 22.04.5 发行版的独立环境安装。 [Windows Triton 兼容矩阵](https://github.com/triton-lang/triton-windows/blob/readme/README.md#3-pytorch) · [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) · [WSL 环境锁](../../../configs/batch_b/s3-wsl-runtime.lock.txt)

## 数值、存档与恢复

- 真实权重加载核对了 426 个文本主干张量。只有官方分类头的 `score.weight` 新初始化。两端均通过 28 维多标签 BCE、右填充和最后内容 token 池化检查，128-token 反向时 248 个 LoRA-B 张量和分类头均收到非零梯度。更新后 LoRA、分类头及 AdamW 状态均发生变化。
- 同一初始 Adapter 和分类头在 48 条 validation、28 个标签上得到配对分数。跨后端概率绝对差均值 **0.00220**，p95 **0.00829**，最大 **0.01811**。固定阈值 0.50 时，1,344 个标签判断中 3 个翻转，涉及 3 条样本。匹配的前 13 次更新中，逐步损失最大绝对差为 0.00469。后续选模、恢复与正式复算应锁定同一后端。
- WSL2 发布包约 248 MiB，包含 496 个 LoRA A/B 张量与完整分类头。另存约 496 MiB 的 AdamW、调度器、游标和 CPU/CUDA/NumPy/Python RNG。独立进程重载后，48 条验证样本分数逐项精确一致，下一次更新的学习率、损失、梯度证据、LoRA、分类头与优化器指纹和 RNG 探针均精确重现。 [恢复存档指纹](acceptance.json) · [本地独立恢复回执](../../../artifacts/batch-b-s3/performance/ubuntu-fused/recovery.json)

## 待审配置和风险

- 候选正式环境为 WSL2。保留同一训练配置和最大长度 64。train/validation 的 43,211 / 5,416 条数据、编码指纹、28 个标签及验证阈值网格已对齐。长度 64 在 train 的覆盖率为 99.9815%，截断 8 条。测试集仅能由通过冻结门禁的后续正式入口读取。 [Tokenizer 审计](tokenization.json)
- 计算成本仍约为一周连续 GPU 时间。建议审阅时以约 160 小时训练加验证为预算基线，并保留早停和同后端恢复。性能窗用相同样本对齐，正式全量运行的时间还会受样本长度、宿主负载和日志写盘影响。
- CUDA 预检验证了短窗和最大长度。为长任务保留显存余量，并记录编译缓存、驱动、空闲内存与实际吞吐。训练及恢复使用相同 WSL 运行时及官方资源指纹。

[Windows 原始采样](../../../artifacts/batch-b-s3/performance/windows-baseline/) · [Windows 分段剖析](../../../artifacts/batch-b-s3/performance/windows-profile-r2/phase-profile.json) · [Windows 匹配遥测](../../../artifacts/batch-b-s3/performance/windows-matched-gpu.csv) · [WSL2 进度记录](../../../artifacts/batch-b-s3/performance/ubuntu-fused/progress.md)
