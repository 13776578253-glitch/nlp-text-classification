# nlp-text-classification

「自然语言处理之文本分类」课程项目。Batch A 在英文 GoEmotions 多标签任务和中文电影评论二分类任务上分别比较 7 个配置。先运行 14 个 seed 42 基础配置，按 validation Macro-F1 完成调参、模型选择和多种子复验，再整批冻结并正式测试。

| 位置 | 内容 |
| --- | --- |
| `src/nlp_text_classification/` | Python 数据加载、指标、训练和 CLI |
| `configs/datasets/` | 两套数据定义 |
| `configs/experiment_suites/batch_a.yaml` | Batch A 实验矩阵 |
| `data/raw/`、`data/processed/` | 本地源数据与派生数据 |
| `artifacts/` | 本地 Run 与批次产物 |
| `docs/` | 架构、契约及执行报告 |

Batch A 已完成正式测试及两轮验证补充实验。Batch B S1 比较预训练 Encoder 的完整微调结果。两批复用冻结数据版本和验证 Macro-F1。

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
nlp-text-classification audit-data --dataset movie_reviews_zh
```

中文审计需要老师提供的三个分区文件及 `wiki_word2vec_50.bin`，默认路径为 `data/raw/movie_reviews_zh/`。官方 GoEmotions 的下载与审计入口，以及正式执行顺序见执行报告。

[文档索引](docs/INDEX.md) · [Batch A 研究归档](docs/finish-task/batch-a/summary.md) · [当前状态](docs/current-task/ACTIVE.md)
