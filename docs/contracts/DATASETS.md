# Batch A 数据契约

| 数据 | 文件与任务 | 分区 |
| --- | --- | --- |
| GoEmotions simplified | `train.tsv`、`dev.tsv`、`test.tsv`、`emotions.txt`。行格式为文本、标签 ID 列表、样本 ID，按 Tab 分列。保留 28 个官方标签与顺序，保留 neutral 的原始共现。 | 官方 train/dev/test 预期 43,410 / 5,426 / 5,427 |
| 中文电影评论 | `train.txt`、`validation.txt`、`test.txt`。行格式为原始 0/1、Tab、预分词文本。以分区与原始行号生成样本 ID，记录并跳过空行。 | 本地审计有效 19,998 / 5,629 / 369 |

源文件放在 `data/raw/<dataset_id>/`。Stage 0 对三分区计算 SHA-256、数量和标签分布，检查 ID 隔离与空白归一化后的完全重复，生成聚合审计。无 ID 的中文数据通过文本审计重合。原始文件按字节保留，派生分区、行 ID 清单及指纹写到 `data/processed/<dataset_id>/<revision>/`。任何排除规则在 Stage 0 登记，应用到同数据集所有模型。

v1 保留完整 test。派生 validation 排除与 test 重合的文本，派生 train 排除与原始 validation/test 重合的文本及集内标签相同的重复项。文本只比较空白归一化的完全匹配。逐行排除理由写入 `excluded.json`。Batch A 训练入口使用 `training.json`，其分区和文件指纹仅包含 train 与 validation。

GoEmotions 英文词级 tokenizer 的规则固定后只用 train 拟合词表、TF-IDF 及 50d 无监督 Word2Vec。中文保留老师的空格切词，并用 train 构建词表，预训练组匹配老师提供的 `wiki_word2vec_50.bin`。两个任务分别保存 tokenizer、词表、OOV/PAD 规则、输入长度、Word2Vec 来源与覆盖率。

中文 v1 原始标签 0 为正向，1 为负向，依据两轮训练样本人工核验固定。二分类适配器将语义正类编码为训练目标 1，并保留到原始 ID 的映射。Stage 3 仅加载派生 train 和 validation。Stage 4 评估入口验证完整冻结清单后核对并读取 test。
