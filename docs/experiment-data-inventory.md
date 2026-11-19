# 实验数据盘点与交付索引

盘点于 2026-11-19，以本地双数据集 v1、Batch A 和 Batch B 的已有冻结版本为准。下面的路径相对于仓库根目录 `D:/Desktop/code/nlp-text-classification`。数据分布来自已有清单和审计汇总。本次核对了两套数据的清单、训练清单、源审计与排除清单指纹，以及 62 个正式测试 Run 的证据路径。

## 1. 数据集与分区

| 数据集 | 口径 | 训练 | 验证 | 测试 | 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| GoEmotions simplified，英文 28 标签 | 原始 | 43,410 | 5,426 | 5,427 | 54,263 |
| GoEmotions simplified，英文 28 标签 | 冻结 v1 | 43,211 | 5,416 | 5,427 | 54,054 |
| 中文电影评论，二分类 | 原始有效样本 | 19,998 | 5,629 | 369 | 25,996 |
| 中文电影评论，二分类 | 冻结 v1 | 19,994 | 5,629 | 369 | 25,992 |
| **双数据集** | **冻结 v1** | **63,205** | **11,045** | **5,796** | **80,046** |

GoEmotions v1 的训练、验证、测试占比分别为 79.94%、10.02%、10.04%，排除训练 199 条、验证 10 条。中文 v1 分别为 76.92%、21.66%、1.42%，排除训练 4 条。中文源文件还有验证 2 个空行、测试 3 个空行，已在原始有效样本计数前跳过。两个测试分区保留全部原始有效样本。 [Stage 0 数据验收](archive/stages/stage0-data-audit.md)

| 文件 | GoEmotions | 中文电影评论 |
| --- | --- | --- |
| 源目录 | [data/raw/goemotions/](../data/raw/goemotions/) | [data/raw/movie_reviews_zh/](../data/raw/movie_reviews_zh/) |
| 源分区 | [train.tsv](../data/raw/goemotions/train.tsv) · [dev.tsv](../data/raw/goemotions/dev.tsv) · [test.tsv](../data/raw/goemotions/test.tsv) | [train.txt](../data/raw/movie_reviews_zh/train.txt) · [validation.txt](../data/raw/movie_reviews_zh/validation.txt) · [test.txt](../data/raw/movie_reviews_zh/test.txt) |
| v1 分区 | [train.jsonl](../data/processed/goemotions/v1/train.jsonl) · [validation.jsonl](../data/processed/goemotions/v1/validation.jsonl) · [test.jsonl](../data/processed/goemotions/v1/test.jsonl) | [train.jsonl](../data/processed/movie_reviews_zh/v1/train.jsonl) · [validation.jsonl](../data/processed/movie_reviews_zh/v1/validation.jsonl) · [test.jsonl](../data/processed/movie_reviews_zh/v1/test.jsonl) |
| 分区与标签指纹 | [dataset_manifest.json](../data/processed/goemotions/v1/dataset_manifest.json) | [dataset_manifest.json](../data/processed/movie_reviews_zh/v1/dataset_manifest.json) |
| 训练入口清单 | [training.json](../data/processed/goemotions/v1/training.json) | [training.json](../data/processed/movie_reviews_zh/v1/training.json) |
| 源分布与排除 | [source_audit.json](../data/processed/goemotions/v1/source_audit.json) · [excluded.json](../data/processed/goemotions/v1/excluded.json) | [source_audit.json](../data/processed/movie_reviews_zh/v1/source_audit.json) · [excluded.json](../data/processed/movie_reviews_zh/v1/excluded.json) |
| 分区配置 | [goemotions.yaml](../configs/datasets/goemotions.yaml) | [movie_reviews_zh.yaml](../configs/datasets/movie_reviews_zh.yaml) |
| 标签与词向量 | [emotions.txt](../data/raw/goemotions/emotions.txt) · [训练得到的 50d Word2Vec](../artifacts/batch-a/embeddings/goemotions-fbd69186f3c8981d-50d-42.kv) | [课程 Word2Vec 50d](../data/raw/movie_reviews_zh/wiki_word2vec_50.bin)，原始 ID 0 正向、1 负向 |

## 2. 全部标签支持数

### GoEmotions 28 标签

同一条文本可对应多个标签。表内各列为标签正例数，整列合计是标签标注次数。冻结 v1 平均每条训练、验证、测试样本分别有 1.178、1.176、1.166 个标签。测试列同时适用于原始和 v1。 [原始分布](../data/processed/goemotions/v1/source_audit.json) · [冻结分布](../data/processed/goemotions/v1/dataset_manifest.json)

| ID | 标签 | 原始训练 | v1 训练 | 原始验证 | v1 验证 | 测试 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | admiration | 4,130 | 4,115 | 488 | 488 | 504 |
| 1 | amusement | 2,328 | 2,325 | 303 | 303 | 264 |
| 2 | anger | 1,567 | 1,564 | 195 | 195 | 198 |
| 3 | annoyance | 2,470 | 2,466 | 303 | 303 | 320 |
| 4 | approval | 2,939 | 2,934 | 397 | 396 | 351 |
| 5 | caring | 1,087 | 1,081 | 153 | 153 | 135 |
| 6 | confusion | 1,368 | 1,363 | 152 | 151 | 153 |
| 7 | curiosity | 2,191 | 2,184 | 248 | 247 | 284 |
| 8 | desire | 641 | 641 | 77 | 77 | 83 |
| 9 | disappointment | 1,269 | 1,266 | 163 | 163 | 151 |
| 10 | disapproval | 2,022 | 2,021 | 292 | 292 | 267 |
| 11 | disgust | 793 | 790 | 97 | 97 | 123 |
| 12 | embarrassment | 303 | 302 | 35 | 35 | 37 |
| 13 | excitement | 853 | 847 | 96 | 96 | 103 |
| 14 | fear | 596 | 594 | 90 | 90 | 78 |
| 15 | gratitude | 2,662 | 2,606 | 358 | 353 | 352 |
| 16 | grief | 77 | 77 | 13 | 13 | 6 |
| 17 | joy | 1,452 | 1,443 | 172 | 172 | 161 |
| 18 | love | 2,086 | 2,057 | 252 | 251 | 238 |
| 19 | nervousness | 164 | 164 | 21 | 21 | 23 |
| 20 | optimism | 1,581 | 1,577 | 209 | 209 | 186 |
| 21 | pride | 111 | 111 | 15 | 15 | 16 |
| 22 | realization | 1,110 | 1,109 | 127 | 127 | 145 |
| 23 | relief | 153 | 153 | 18 | 18 | 11 |
| 24 | remorse | 545 | 544 | 68 | 68 | 56 |
| 25 | sadness | 1,326 | 1,322 | 143 | 143 | 156 |
| 26 | surprise | 1,060 | 1,057 | 129 | 129 | 141 |
| 27 | neutral | 14,219 | 14,176 | 1,766 | 1,765 | 1,787 |

### 中文电影评论

| 类别 | 源 ID | 训练目标 | 原始训练 | v1 训练 | v1 验证 | v1 测试 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| negative | 1 | 0 | 9,999 | 9,997 | 2,812 | 187 |
| positive | 0 | 1 | 9,999 | 9,997 | 2,817 | 182 |

原始有效验证、测试的逐类数与 v1 一致。`dataset_manifest.json` 的标签顺序是 `negative, positive`，对应源 ID `[1, 0]`。Batch B B4 使用两维 Softmax，第二列为语义正类概率。 [标签契约](contracts/DATASETS.md) · [Encoder 测试验收](finish-task/batch-b-final.md)

### 长度口径

Batch A 的 v1 训练长度 P50/P95/P99 分别为 GoEmotions 15/28/32、中文 43/62/68，训练长度上限分别为 128 和 256。Encoder 的 B1/B2/B3/B4 使用各自 Tokenizer，冻结上限为 64/64/64/128，训练截断分别为 3/12/12/112 条，验证截断分别为 0/2/2/40 条。Qwen 上限为 64，训练、验证、测试分别截断 8、1、1 条。 [Batch A 长度审计](archive/stages/stage0-data-audit.md) · [Encoder Tokenization 清单](../configs/batch_b/tokenization_lock.json) · [Qwen 训练和验证 Tokenization](../reports/batch-b-s3/performance/tokenization.json) · [Qwen 测试 Tokenization](../reports/batch-b-s4/formal/test-tokenization.json)

## 3. 结果、检查点与正式测试证据

| 阶段 | 队列与指标口径 | 汇总、冻结和本地证据入口 |
| --- | --- | --- |
| Batch A 正式 | 双任务 14 个主配置。47 个正式测试 Run，含 14 主 Run、30 必选复验、3 追加复验。英文 22、中文 25。 | [逐 Run 索引](finish-task/batch-a/model-index.json) · [冻结清单](../artifacts/batch-a/final_runs.yaml) · [逐类指标](../reports/batch-a/255a9bfe53ee/class_metrics.json) · [报告](../reports/batch-a/255a9bfe53ee/report.md) |
| Batch A Supplement S1 | GoEmotions 验证，6 个 24 轮主配置、12 个跨种子复验，另有 6 个前 12 轮重放。 | [结果](../reports/batch-a-s1/results.json) · [本地 Runs](../artifacts/batch-a-s1/runs/) · [验收](../artifacts/batch-a-s1/delivery_audit.json) |
| Batch A Supplement S2 | GoEmotions 验证，Vanilla RNN 3 个续训、LSTM 1 个续训。 | [结果](../reports/batch-a-s2/results.json) · [本地 Runs](../artifacts/batch-a-s2/runs/) · [验收](../artifacts/batch-a-s2/delivery_audit.json) |
| Batch B S1 Encoder | 两套数据验证，B1–B4 共 8 个训练槽位。 | [验证结果](../reports/batch-b-s1/results.json) · [本地 Runs](../artifacts/batch-b-s1/formal/runs/) · [模型资源](../configs/batch_b/resource_lock.json) |
| Batch B S2 Encoder | 两套数据验证，9 个新增尝试，其中 8 个复验和 1 个 B1 长预算探针。最终 12 个验证入选 Run 与 Encoder 正式测试对应。 | [验证结果](../reports/batch-b-s2/results.json) · [逐类验证](../reports/batch-b-s2/label-metrics.json) · [本地 Runs](../artifacts/batch-b-s2/runs/) |
| Batch B Encoder 正式 | B1–B3 英文 9 个 Run，B4 中文 3 个 Run，统一 3 个 Seed。 | [测试结果](../reports/batch-b-final/2605f6766085/results.json) · [逐类测试](../reports/batch-b-final/2605f6766085/label-metrics.json) · [冻结清单](../reports/batch-b-final/2605f6766085/freeze.json) · [本地测试证据](../artifacts/batch-b-final/test/2605f6766085ab675d97de6a48003f1a38ee78174807d0c2e0bc8861b9cbd6f1/) |
| Batch B S3 Qwen | 工程预检、量化加载与恢复、Windows/WSL2 性能证据。 | [性能报告](../reports/batch-b-s3/performance/report.md) · [官方资源清单](../reports/batch-b-s3/performance/model-resources.json) · [官方基座快照](../artifacts/batch-b-s3/official/1001bb4d826a52d1f399e183466143f4da7b741b/) |
| Batch B S4 Qwen | GoEmotions Seed 42、123、2026，三组独立训练、验证冻结、正式测试。 | [验证审计与轨迹](../reports/batch-b-s4/formal/freeze-audit.json) · [冻结引用](../reports/batch-b-s4/formal/freeze-ref.json) · [测试指标与签收](../reports/batch-b-s4/formal/test-acceptance.json) · [本地 Runs](../artifacts/batch-b-s4/formal/runs/) · [本地测试证据](../artifacts/batch-b-s4/formal/test/97ce29261f3844e715731f62f7859df2397db2e7381744bc8a0db86cff4e5d15/) |

62 个正式测试 Run 中，英文 34、中文 28。按 Run 累计有 194,850 条逐样本测试记录，三个入口共享各自任务的固定测试分区。验证补充、前缀重放、HPO Trial、性能短测和探索性预检均有各自记录。

- Batch A 的实际 Run 目录是 `artifacts/runs/<dataset>/<model>/<run_id>/`。逐 Run 索引给出精确路径。正式证据位于每个 Run 的 `test/`，包括 `predictions_test.jsonl`、`metrics_test.json`、`evaluation_manifest.json`。模型在 `checkpoint/`，E1 为 `model.joblib`，神经模型为 `model.pt`。 [逐 Run 索引](finish-task/batch-a/model-index.json)
- Encoder 的正式测试证据以 `<model>-<seed>/` 存放，每个目录有 `predictions_test.jsonl`、`metrics_test.json` 和 `evaluation_manifest.json`。训练 Run 路径和 `best-epoch-*.pt` 由冻结清单定位。离线基座和 Tokenizer 缓存在 [artifacts/batch-b-s1/hf-cache/](../artifacts/batch-b-s1/hf-cache/)。 [Encoder 冻结清单](../reports/batch-b-final/2605f6766085/freeze.json)
- Qwen Seed 42/123 的最佳步为 `step-00005402`，Seed 2026 为 `step-00002701`。在 `artifacts/batch-b-s4/formal/runs/seed-<seed>/checkpoints/<step>/adapter/` 查找 `adapter_model.safetensors` 和 `adapter_config.json`。各 Seed 测试目录含 `predictions_test.jsonl`、`metrics_test.json` 和 `evaluation.json`。 [Qwen 冻结清单](../reports/batch-b-s4/formal/frozen/97ce29261f3844e715731f62f7859df2397db2e7381744bc8a0db86cff4e5d15.json)

## 4. 前端与报告取数

| 场景 | 文件与字段 |
| --- | --- |
| 数据集概览、分区、类别分布 | 两份 `dataset_manifest.json` 的 `splits`、`label_names`、`label_ids`、`derived_summary.label_counts`。原始对照取 `source_audit.json`。 |
| 模型排行榜、复验与逐类图 | Batch A `model-index.json → formal.runs`、`class_metrics.json`。Encoder 正式 `results.json → runs / three_seed_summary`、`label-metrics.json`。Qwen `test-acceptance.json → runs / aggregate / per_label`。按数据集和验证/测试阶段展示。 |
| 曲线与报告 | [Batch A 曲线](../reports/batch-a/255a9bfe53ee/curves/) · [Batch A LaTeX](../reports/batch-a/255a9bfe53ee/report.tex) · [Encoder 正式报告](../reports/batch-b-final/2605f6766085/report.md) · [Qwen 正式报告](../reports/batch-b-s4/formal/final-test.md) · [Qwen 曲线](../reports/batch-b-s4/formal/training-curves.png) · [Qwen LaTeX 表](../reports/batch-b-s4/formal/test-results.tex) |
| 前端与推理入口 | [React/Vite 配置](../apps/web/package.json)，`main.tsx` 与 `style.css` 当前为空。[FastAPI](../apps/api/main.py) 为 E1 演示入口，当前按旧的扁平 Run 路径查找 `model.joblib`。接入现有正式模型时先对齐 Run 索引和检查点路径。[S4 只读训练监控](../apps/monitor/) 已有独立页面。 |

冻结归档时，公开展示包使用标签、分区统计、汇总指标和曲线。本地证据包保存原始及 v1 数据、逐样本预测、检查点、Tokenizer、配置和日志。仓库的 `data/` 与 `artifacts/` 内容受 Git 忽略规则管理，需要随本地归档单独保存。此清单用于准备总归档，现有各阶段冻结版本继续作为证据入口。
