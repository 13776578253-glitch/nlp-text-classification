# EmotionLab 系统架构

## 目录结构

```text
apps/
  api/                         # FastAPI 在线推理服务（统一加载 Batch A / Encoder 并转发 Qwen Worker）
  monitor/                     # S4 Qwen 只读训练监控页面
  web/                         # EmotionLab 三章交互式展示与推理前端（React + TypeScript + Vite）
configs/
  datasets/                    # 双数据集定义（goemotions / movie_reviews_zh）
  experiment_suites/           # Batch A 实验矩阵
  batch_b/                     # B1–B4 配置、资源及 Tokenization 规范
data/
  raw/                         # 本地原始数据与中文 wiki_word2vec_50.bin（.gitignore 屏蔽）
  processed/                   # 本地清洗后标准分区、训练清单与源审计 JSON（.gitignore 屏蔽）
docs/
  contracts/                   # 数据、实验、评估与产物契约
  finish-task/                 # Batch A/B 实验总结与前端交接说明
  archive/                     # 阶段执行计划与运行手册
reports/                       # 各阶段正式指标 JSON、Markdown/LaTeX 报告与图表
scripts/                       # 前端数据导出 export_web_data.py、全量校验与 WSL2 Qwen 脚本
src/nlp_text_classification/
  cli.py                       # 统一 CLI 入口
  data/                        # 分区加载、源审计与 Transformer 输入构造
  models/                      # TextCNN、RNN/LSTM/BiLSTM 模型定义
  training/                    # 通用训练器与 Transformer 训练器
  experiments/                 # Batch A/B 队列调度、S1–S4 与评估编排
  evaluation/                  # 多标签指标、Bootstrap 显著性检验与前端数据导出
  inference/                   # 模型注册表 registry、多架构适配器 adapters 与 Qwen Worker
tests/
  unit/                        # 22 项单元测试（数据、训练、导出、推理、监控）
  integration/                 # 2 项集成测试（断点恢复、E1 冒烟）
artifacts/                     # 本地模型权重、预测结果与实验记录（.gitignore 屏蔽）
```

Python 包与 CLI 使用 `nlp_text_classification` 与 `nlp-text-classification`。源数据默认读取 `data/raw/<dataset_id>/`，自定义路径可通过 `--data-dir` 或环境变量 `NLP_TEXT_CLASSIFICATION_DATA` 指定。

## 核心链路

1. **数据准备与审计**：`data/loaders/` 解析双数据集标准分区，`data/audit.py` 输出聚合统计、来源文件哈希、排除样本与清洗清单。
2. **模型训练与调优**：`experiments/` 将实验矩阵展开为执行队列。Batch A 覆盖 E1–E5 并完成 HPO 与多种子复验；Batch B S1/S2 完成 B1–B4 四组预训练 Encoder 微调；Batch B S3/S4 在 WSL2 QLoRA 环境下完成 Qwen3.5-4B 三种子训练。
3. **统一评测与统计检验**：基于验证集 `Macro-F1` 选定最优配置与阈值后，在独立测试集上一次性计算最终指标、逐类 F1 与逐样本预测，输出至 `reports/`，并由 `evaluation/dashboard_comparisons.py` 执行 100,000 次分层配对 Bootstrap 检验。
4. **前端展示与在线推理**：`scripts/export_web_data.py` 将各阶段结果聚合成 `apps/web/src/data/dashboard-data.json`；`apps/api/main.py` 通过 `inference/registry.py` 与 `adapters.py` 提供本地多模型实时推理，并将 Qwen 推理请求转发至 WSL2 Worker（`inference/qwen_worker.py`）。

[文档索引](INDEX.md) · [实验数据盘点](experiment-data-inventory.md) · [前端与推理交接](finish-task/frontend-freeze.md)
