# Stage 0 数据冻结验收

| 数据 | 原 train / validation / test | 派生 v1 train / validation / test |
| --- | --- | --- |
| GoEmotions | 43,410 / 5,426 / 5,427 | 43,211 / 5,416 / 5,427 |
| 中文电影评论 | 19,998 / 5,629 / 369 | 19,994 / 5,629 / 369 |

GoEmotions 四个下载文件的 Git blob SHA 与 Google Research 官方目录 API 一致，28 个标签和分区计数通过。训练排除了 80 条同测试文本、47 条同原验证文本、72 条本分区同标签重复行。验证排除了 10 条同测试文本。两个完整测试集与原始划分一致。

中文源数据和向量来自课程 ZIP。训练排除 2 条同验证文本与 2 条训练内同标签重复行。人工复核两轮训练抽样，每类分别为 8 条和 12 条。第二轮候选为 25–65 词的训练行，原始标签 0 与 1 分别用 seed 2026 和 2027 抽取，评论整体倾向支持原始 0 表示正向、1 表示负向，混合情绪样本保留原标签。

GoEmotions 派生训练 P50/P95/P99 为 15/28/32 词元，保留长度 128。中文为 43/62/68，保留 256。两个数据集派生版本的三对分区空白归一化文本交集均为零。文件、分区和审计指纹已复核。

本地原始证据位于 `data/processed/<dataset>/v1/`，包含三个 JSONL 分区、`excluded.json`、`source_audit.json`、`training.json` 和 `dataset_manifest.json`。配置固定两个 manifest 的 SHA-256。

Stage 0 检查使用 `freeze-data --dataset goemotions --revision v1` 和 `freeze-data --dataset movie_reviews_zh --revision v1 --positive-id 0` 生成证据。9 项 pytest 和 Ruff 通过。
