# Batch B S1 · 官方模型资源验收

四个仓库均为 Apache-2.0。已按 Commit SHA 下载到 `artifacts/batch-b-s1/hf-cache/B1–B4/` 下各自的 revision 目录。锁文件记录每个文件的长度、上游 Git blob 或 LFS SHA-256、实测 SHA-256、依赖版本和离线加载结果。预训练权重保留在本机缓存。

| 配置 | Commit SHA | 权重字节 | 分类模型参数 | 离线输出 |
| --- | --- | ---: | ---: | --- |
| B1 | `12040accade4e8a0f71eabdb258fecc2e7e948be` | 267,954,768 | 66,975,004 | 2 × 28 |
| B2 | `8949b909ec900327062f0ebf497f51aef5e6f0c8` | 598,635,032 | 149,626,396 | 2 × 28 |
| B3 | `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | 1,583,544,840 | 395,859,996 | 2 × 28 |
| B4 | `a986e004d2a7f2a1c2f5a3edef4e20604a974ed1` | 411,582,858 | 102,269,186 | 2 × 2 |

本地环境使用 PyTorch `2.9.1+cu128` 与 Transformers 4.57.6，以 `local_files_only=True`、`trust_remote_code=False` 加载。B1–B3 使用 safetensors。B4 使用官方 `pytorch_model.bin` 和 `weights_only=True`，已核对 LFS 哈希。由预训练掩码模型创建任务分类头，逐模型核验主干加载和分类输出。MacBERT 元数据首次请求因 TLS 断开失败，重试后同一官方仓库通过校验。原始日志位于 `artifacts/batch-b-s1/resources-terminal.log`。

[资源锁文件](../../../configs/batch_b/resource_lock.json) · [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) · [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) · [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) · [Chinese MacBERT-base](https://huggingface.co/hfl/chinese-macbert-base)
