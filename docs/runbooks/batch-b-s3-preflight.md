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
