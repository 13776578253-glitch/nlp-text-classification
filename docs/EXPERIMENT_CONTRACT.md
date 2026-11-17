# nlp-text-classification 实验契约（v1）

## 数据契约

- `task_id=goemotions/simplified/multilabel-v1`
- 官方 TSV：`train.tsv`（43,410）、`dev.tsv`（5,426）、`test.tsv`（5,427）。
- 字段顺序：`text<TAB>comma-separated-label-ids<TAB>id`，无表头。
- `emotions.txt` 第 0–27 行对应 28 标签，最后一个为 `neutral`；数据转换时不重排标签、不默认修改 neutral 的共现情况。
- train/validation/test **保持官方划分**，检查样本 ID 无交叉；相同/近似文本跨分区污染另做审计。
- 文本中含 Reddit 风格缩写、脏话、潜在偏见内容，仅作研究数据处理。
- 不从测试集拟合 TF-IDF、词表、Word2Vec 或调整任何阈值。

## 训练与评估

- 每次训练固定随机种子，并记录 Python、关键库版本、Git Commit、数据校验值、模型名、配置、训练时间（后续实验加入）。
- **主指标：** Macro-F1、Micro-F1。辅助：Samples-F1、Hamming Loss、Subset Accuracy、按标签 Precision/Recall/F1、PR-AUC（后续实现）。
- 模型输出 28 个分数。多标签 neural models：sigmoid(logits)，BCEWithLogitsLoss；SVM：decision_function margin。
- 分数类型必须显式标注为 `probability`（未校准 sigmoid 分数）或 `decision`（SVM margin），不得混为概率。
- **阈值只在验证集选择。** 初版使用全局阈值，以 Validation Micro-F1 最优为原则，并在报告记录。
- 各模型统一使用相同划分。对比尽量保持近似训练预算；明显不同的 tokenizer / 优化器 / 预训练语料需要在报告说明。
- 训练中按 Validation 选择最佳 Epoch；完成后选择最终 Run，冻结配置后进行最终 test，记录分析过程。
- 优先记录每类 support，防止常见 neutral/joy 等标签掩盖稀有标签效果。

## 标准运行产物（逐步实施）

```text
artifacts/runs/<run_id>/
  run_manifest.json          # 提交、dataset、model、seed、环境、配置指纹
  config.json                # 完整已解析配置
  metrics_validation.json    # 训练期可反复查看
  thresholds.json            # 验证集选定阈值和分数语义
  metrics_test.json          # 仅 final-test 阶段生成
  predictions_validation.jsonl
  predictions_test.jsonl    # 仅 final-test 阶段生成
  model.*                    # 模型权重 + tokenizer / 词表
  history.csv                # 深度学习训练过程
```

首个 E1 实现已经包含 manifest、config、validation 指标、thresholds、逐样本预测与 model；后续深度学习实验将补充 history 和模型专属 checkpoint 元数据。

## E5 消融规则

TextCNN 结构、词表、Embedding 维度、序列长度、训练数据和预算相同。E2 随机 Embedding 直接作为 E5-A；E5-B/C 的 Word2Vec 使用相同训练集训练得到的向量。E5-C 断言冻结层权重在训练后未变化。

## 模型选择安全线

- 先选定 E1 Run 并固定模型、配置与验证集阈值，再运行 `nlp-text-classification evaluate-final-e1`。命令核对数据文件哈希，遇到已有测试指标的 Run 会退出。报告须标注所用 Run ID。
- 禁止观察测试指标后反复手动改超参数并将其作为同一份盲测结果。
- API 加载 `joblib` 产物仅限可信本地文件，不允许任意上传后执行反序列化。
