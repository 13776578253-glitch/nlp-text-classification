# Batch B S3 · Qwen3.5-4B 工程预检

S3 使用官方 Qwen3.5-4B-Base 的文本分类类、NF4 双重量化、248 个线性投影的 Rank 32 / Alpha 64 LoRA，以及完整 28 维 `score` 分类头。[配置](../../configs/experiment_suites/batch_b_s3.yaml) 锁定模型 Revision、标签任务、LoRA、优化器分组和验证阈值。预检接受冻结的 train/validation 数据与合成输入。

## 依赖与资源

使用独立 Python 3.12 环境，PyTorch 2.9.1+cu128、Transformers 5.10.2、PEFT 0.19.0、bitsandbytes 0.49.2、Accelerate 1.16.0。安装后执行 `uv pip check`，版本见 [S3 环境锁](../../configs/batch_b/s3-runtime.lock.txt)。可在隔离目录复用同版本已安装的 CUDA Torch 文件副本。由独立入口下载官方 safetensors、Config、Tokenizer 和许可证，并逐个校对 LFS SHA-256 或 Git blob。

Windows 上默认 mmap 读取权重曾触发原生访问异常。正式配置使用经过边界校验的分段读取，将官方文本张量映射到原生分类模型，直接通过 bitsandbytes 生成 NF4。原分片保持原样。加载前核对两个 safetensors 文件头与索引，加载后核对模块、位置及抽样原值和量化结果。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:HF_HOME = Join-Path (Get-Location) 'artifacts/batch-b-s3/hf-download-cache'
$s3python = 'artifacts/batch-b-s3/runtime/Scripts/python.exe'
& $s3python -u -m nlp_text_classification.experiments.batch_b_s3_resources
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& $s3python -u -m nlp_text_classification.experiments.batch_b_s3 kernel
& $s3python -u -m nlp_text_classification.experiments.batch_b_s3 tokens
& $s3python -u -m nlp_text_classification.experiments.batch_b_s3 benchmark --attempt preflight
& $s3python -u -m nlp_text_classification.experiments.batch_b_s3 recover --attempt preflight
& $s3python -u -m nlp_text_classification.experiments.batch_b_s3 report --attempt preflight
```

每个失败或重试使用独立 attempt。下载可续传同一官方 Revision。CUDA 使用共享 GPU 锁，正式环境的 Python、缓存和 Checkpoint 分别隔离。资源门槛以启动 8 GiB 主存、运行 6 GiB 主存和计划 2 GiB GPU 余量作为预检条件。日志位于 `artifacts/batch-b-s3/<attempt>/progress.md` 和 `progress.jsonl`。

## 验收与正式入口

- `model-resources.json` 对应官方 Revision 与全部资源指纹。`load-audit.json` 核对 426 个源张量的形状、预期缺失的 `score.weight` 和 NF4 抽样。 `lora-audit.json` 列全部命中路径、张量形状和实际可训练参数。
- `tokenization.json` 从 train 分布锁定长度，validation 记录同一规则的覆盖和编码指纹。`functional.json` 验证异长右填充、PAD=EOS 反例、池化、BCE、28 维输出和最大长度反向。
- `updates.jsonl` 记录预热和 100 次稳定更新的损失、梯度、双学习率与耗时。`benchmark.json` 汇总内存、吞吐量、估算预算和发布 Checkpoint 指纹。耗时估算以同一环境的实际样本吞吐为基准，验证、存盘、恢复和其他任务占用另计。
- Adapter safetensors 包含全部 LoRA 张量及完整分类头。新进程校验验证分数、阈值和预测，再恢复 AdamW、调度器、RNG 和数据游标，比较下一步的模型与优化器指纹。 `recovery.json` 记录验收。

预检完成后提交解析过的正式配置和风险清单，等待正式训练审阅。三种子训练及正式 test 属于后续阶段。

## WSL2 性能对照

从同一工作区调用下列命令。Windows 入口生成两边共用的初始 Adapter、分类头与 RNG，随后在现有 Ubuntu 发行版中使用独立的 ext4 环境。权重和数据均读取本地原路径。仅一种后端占用 GPU。

```powershell
& $s3python -u -m nlp_text_classification.experiments.batch_b_s3_perf fallback --attempt windows-profile-new
wsl.exe -d Ubuntu-22.04 --cd /mnt/d/Desktop/code/nlp-text-classification --exec bash scripts/batch_b_s3_wsl_setup.sh
wsl.exe -d Ubuntu-22.04 --cd /mnt/d/Desktop/code/nlp-text-classification --exec bash scripts/batch_b_s3_wsl_perf.sh --attempt ubuntu-fused-new
wsl.exe -d Ubuntu-22.04 --cd /mnt/d/Desktop/code/nlp-text-classification --exec bash scripts/batch_b_s3_wsl_perf.sh --attempt ubuntu-fused-new --recover
```

匹配训练样本顺序、micro-batch 1、累积 16 和 5 次预热。稳态比较用第 6–13 次更新，性能剖析用第 5 次更新，CUDA event 记录数据准备、前向、反向、重计算、梯度裁剪和 AdamW。重计算已包含在反向跨度中。Windows 补采的匹配遥测可用 `fallback --attempt <new-attempt> --matched-telemetry`，预热后测第 6–7 次更新。依赖锁和验收见 [WSL 环境锁](../../configs/batch_b/s3-wsl-runtime.lock.txt) 与 [性能报告](../../reports/batch-b-s3/performance/report.md)。
