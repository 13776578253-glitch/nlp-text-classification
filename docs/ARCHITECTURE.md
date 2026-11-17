# nlp-text-classification 架构

仓库将实验训练、模型推理和演示界面放在同一个 Git 项目中。CLI 负责下载数据、训练与评估，FastAPI 从本地 Run 加载模型，React 负责输入文本和展示预测。

```text
CLI → data + experiments → evaluation + training → artifacts/runs
React → FastAPI → E1 预测 → artifacts/runs
artifacts/runs → 图表与表格（计划）→ LaTeX 报告
```

| 位置 | 当前职责 |
| --- | --- |
| `src/nlp_text_classification/cli.py` | 数据下载和校验、训练 E1、最终测试、单文本推理 |
| `src/nlp_text_classification/data/` | GoEmotions 下载、TSV 解析、标签和划分校验 |
| `src/nlp_text_classification/evaluation/` | 多标签指标和验证集阈值搜索 |
| `src/nlp_text_classification/experiments/` | E1 向量化、训练、保存和推理 |
| `src/nlp_text_classification/training/` | Run 目录、Git 版本与 JSON 产物 |
| `apps/api/` | `/api/health`、`/api/labels`、`/api/runs`、`/api/predict` |
| `apps/web/` | React/Vite 文本分类演示 |
| `configs/` | 数据与 E1–E5 实验配置，记录计划参数 |
| `data/`、`artifacts/` | 本地数据与训练产物 |
| `tests/`、`reports/` | 自动化测试和 LaTeX 报告骨架 |

Python 分发包和命令使用 `nlp-text-classification`，导入路径使用 `nlp_text_classification`。API 默认从 `artifacts/runs/` 查找可信的训练产物，可通过 `NLP_TEXT_CLASSIFICATION_ARTIFACTS` 指定产物根目录，通过 `NLP_TEXT_CLASSIFICATION_DATA` 指定数据目录，并通过 `NLP_TEXT_CLASSIFICATION_RUN_ID` 配置默认模型。

后续先在 `training/` 增加 E2、E3 和 E5 共用的 PyTorch 训练流程，再接入 Transformer 训练。当前 API 直接调用 E1 推理，后续由 `inference/` 提供统一的多模型加载和分数契约。报告计划从冻结的 Run 读取指标并生成图表与 LaTeX 表格。20 Newsgroups 使用独立的单标签数据与指标协议。
