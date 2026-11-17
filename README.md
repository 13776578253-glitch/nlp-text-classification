# nlp-text-classification

这是「自然语言处理之文本分类」课程项目，目标是训练、评估和演示文本分类模型。主实验使用 GoEmotions，比较统计学习、深度学习和预训练模型在英文短文本多标签分类上的表现。课程收尾时使用 20 Newsgroups 做一组轻量的单标签新闻主题分类对照。

**当前进度**：E1 已实现，并在合成数据上验证了训练、指标计算、模型重载与一次性最终测试流程。正式 GoEmotions 训练、E2–E5 和浏览器联调待完成。下一步是在官方数据上完成 E1 训练与验证。

## 实验规划

| 编号 | 方法 | 验证目标 |
| --- | --- | --- |
| E1 | TF-IDF + One-vs-Rest Linear SVM | 建立统计学习基线 |
| E2 | TextCNN + 随机可训练词嵌入 | 衡量卷积提取局部文本特征的效果 |
| E3 | BiLSTM + 多标签分类头 | 衡量双向序列建模的效果 |
| E4 | DistilBERT、ModernBERT-base，ModernBERT-large 为增强项 | 比较预训练编码器的效果与训练成本 |
| E5 | TextCNN 的 Word2Vec 实验，含微调与冻结 | 与 E2 的随机初始化结果对照，量化预训练词嵌入和冻结策略的影响 |

GoEmotions 使用官方简化版。训练、验证、测试分别为 43,410、5,426、5,427 条，共 28 个标签（含 Neutral）。各实验共用官方划分和标签顺序，在训练集上拟合，在验证集上选模型和分类阈值，冻结所选 Run 后执行最终测试。主指标为 Micro-F1 和 Macro-F1，并记录逐标签结果。[数据来源](https://github.com/google-research/google-research/tree/master/goemotions) · [论文](https://aclanthology.org/2020.acl-main.372/) · [实验契约](docs/EXPERIMENT_CONTRACT.md)

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`src/nlp_text_classification/`](src/nlp_text_classification/) | Python 核心代码，包含数据加载、E1、评价指标、Run 管理和 CLI |
| [`configs/`](configs/) | 数据与 E1–E5 实验配置，E1 CLI 当前使用与 YAML 一致的默认值 |
| [`apps/api/`](apps/api/) · [`apps/web/`](apps/web/) | FastAPI 推理接口和 React/Vite 演示原型 |
| [`tests/`](tests/) · [`.github/workflows/`](.github/workflows/) | 单元测试、合成数据集成测试和 GitHub Actions |
| `data/` · `artifacts/` | 本地下载数据、模型权重、指标和逐样本预测，生成内容由 Git 忽略 |
| [`reports/`](reports/) · [`docs/`](docs/) | LaTeX 报告框架、实验协议和实施计划 |

## 从零运行 E1

以下命令在仓库根目录运行。需要 Python 3.10 或更新版本，推荐 3.11。下载数据需要能访问 GitHub Raw。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,api]"
python -m pytest -q

nlp-text-classification download
nlp-text-classification validate
nlp-text-classification train-e1
```

在 macOS 或 Linux 上，用 `python3 -m venv .venv` 创建环境，再执行 `source .venv/bin/activate`。其余命令相同。

训练命令会输出 Run 路径，可在目录内查看 `metrics_validation.json`、`thresholds.json` 和 `run_manifest.json`。用实际 Run 路径替换下方的占位值。

```powershell
nlp-text-classification predict --run-dir "artifacts/runs/<RUN_ID>" --text "I am excited and grateful."
# 选定并冻结 Run 后，执行一次最终测试。
nlp-text-classification evaluate-final-e1 --run-dir "artifacts/runs/<RUN_ID>"
```

## 启动演示

训练 E1 后，在仓库根目录启动 API。若要为直接调用 API 指定默认 Run，将下面的占位值替换为实际 Run ID。

```powershell
$env:NLP_TEXT_CLASSIFICATION_RUN_ID = "<RUN_ID>"
uvicorn apps.api.main:app --reload --port 8000
```

另开一个终端，使用 Node.js 20+ 启动 Web。跟随 Vite 输出打开本地地址，页面通过 `/api` 请求后端，并允许选择已有训练产物。

```powershell
cd apps/web
npm install
npm run dev
```

补充文档：[课程需求](docs/PROJECT_REQUIREMENTS.md) · [架构](docs/ARCHITECTURE.md) · [当前状态](docs/IMPLEMENTATION_STATUS.md) · [阶段路线](docs/ROADMAP.md) · [GitHub 协作](docs/GITHUB_PUSH.md) · [LaTeX 报告](reports/paper/main.tex)
