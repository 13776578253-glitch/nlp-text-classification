# Batch A V2 架构

## 当前文件树

```text
apps/
  api/                         # 现有 GoEmotions E1 本地推理服务
  web/                         # 原有构建配置，前端源码为空
configs/
  datasets/                    # 两种任务各一份数据定义
  experiment_suites/batch_a.yaml
  batch_b/                     # 独立的英文 Transformer 草案
data/
  raw/movie_reviews_zh/        # 本地导入的原始中文数据
  processed/movie_reviews_zh/  # 本地聚合审计 JSON
docs/
  contracts/
  current-task/
  plans/
src/nlp_text_classification/
  cli.py
  data/
    contracts.py               # 统一样本字段
    audit.py                   # 哈希、数量、类别与重复审计
    download.py                # GoEmotions 来源获取
    loaders/                   # 各自解析固定分区
      goemotions.py
      movie_reviews_zh.py
  evaluation/multilabel.py
  experiments/e1_svm.py
  training/run.py
tests/
  unit/
  integration/
artifacts/                     # 本地输出
```

Python 包和 CLI 继续使用 `nlp_text_classification` 和 `nlp-text-classification`。原 `data.goemotions` 导入路径转发到新 loader。源数据默认用 `data/raw/<dataset_id>/`，历史 GoEmotions 路径可通过 `--data-dir` 指定。API 用 `NLP_TEXT_CLASSIFICATION_DATA` 指定数据路径，当前仍用现有 E1 Run 格式推理。

## 执行依赖

- **Stage 0** 的数据审计独立运行。loader 解析固定分区。audit 读取三分区，输出聚合统计、来源文件哈希和清洗前证据。
- **Stage 1–3** 的配置解析器将 suite 展开成 14 个基础配置。任务适配器将训练与验证转成单 logit 二分类或 28 logit 多标签目标，提供 train-only 的词表及特征。模型模块只接受张量并返回分数。训练器统一管理训练、验证、最优 checkpoint、配置快照和 Run 生命周期。批次队列用相同入口执行基础训练、Optuna Trial、受控补跑及复验。
- **Stage 4** 的独立审计和冻结校验器检查 14 个主 Run、入选复验和产物指纹。最终评估器单独获得 test 路径，以冻结模型和阈值计算正式结果，禁止覆盖成功输出。
- **Stage 5** 的汇总器从冻结清单及正式指标生成验收文件与课程报告。汇总器读取现有产物，推理服务只加载可信本地模型。前端在正式测试结束后另行开展。

当前落地到 loader、独立数据审计与 E1 训练边界。配置解析器、共享训练器、二分类指标、模型模块和整批冻结评估按后续阶段引入，对应模块随实现创建。

[数据契约](contracts/DATASETS.md) · [评估契约](contracts/EVALUATION.md) · [产物契约](contracts/ARTIFACTS.md)
