# 前端冻结交接

2026-11-20 · `main`

## 冻结范围

| 页面 | 已验收行为 |
| --- | --- |
| 01 推理测试 | 按 SVM、TextCNN 及其词向量变体、RNN/LSTM/BiLSTM、BERT 系、Qwen 排列。展示名称与后端模型 ID 分离。推理后上滑回输入，下滑可查看原结果，点击返回输入按钮会清除结果。 |
| 02 模型能力 | 推理延迟与训练时长各有散点图和可排序排行表。训练时长与 Test Macro-F1 取同一 Seed 42 主运行。长尾页展示 GoEmotions 28 类 F1。Bootstrap 分为架构演进和受控消融。 |
| 03 训练数据 | 两套数据集的分区概览、GoEmotions 类别分布、单模型与多模型训练曲线。SVM 展示单次拟合的验证和测试分数。训练曲线支持选择已有 Seed，多模型对比按 Loss 函数分图。 |

模型展示名与简写维护在 [inference-models.ts](../../apps/web/src/data/inference-models.ts)。前端入口是 [App.tsx](../../apps/web/src/App.tsx)。静态评测在 [dashboard-data.json](../../apps/web/src/data/dashboard-data.json)，可通过 [导出脚本](../../scripts/export_web_data.py) 从本地冻结产物重建。冻结 JSON 的 SHA-256 为 `8526eaeda85f98fc9c47ba78553394f1b78c809e715da90c3fc9fa4b6444c7a7`。

## 指标口径

- 数据快照覆盖 80,046 条冻结样本和 62 个正式测试 Run。194,850 是按 Run 累计的预测条数。训练收敛单模型视图列出 51 个模型与 Seed 组合，其中 45 个有逐 Epoch 曲线，6 个是 SVM 单次拟合。
- 速度视图优先使用可用的三 Seed Test Macro-F1 均值。训练时长取 Seed 42 主运行日志，包含验证耗时，纵轴使用同一运行的 Test Macro-F1。
- Bootstrap 使用中文电影评论冻结测试集 369 条样本，按真实类别分层、配对重采样 100,000 次。七组比较统一做 Bonferroni 校正，双侧单对区间为 99.286%，总体显著性水平为 0.05。架构演进四组、消融三组。BiLSTM − LSTM 和 MacBERT − BiLSTM 标记为探索性比较。 [分析实现](../../src/nlp_text_classification/evaluation/dashboard_comparisons.py)
- 英文与中文的训练、验证和测试分区，以及 Batch A、Encoder、Qwen 的冻结引用，以 [实验数据盘点](../experiment-data-inventory.md) 和各批次原始清单为准。

## 本地运行与验收

从仓库根目录分别启动已安装依赖的 Windows FastAPI 和 Vite。

```powershell
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
npm --prefix apps/web run dev -- --host 127.0.0.1
```

Qwen 使用独立的 WSL2 环境与本地模型缓存。在 WSL 中从同一仓库根目录启动。

```bash
bash scripts/start_qwen_worker_wsl.sh --preload-seed 42
```

- 页面入口为 [本地演示](http://127.0.0.1:5173/)。冻结时 Vite 和 FastAPI 均在线，Qwen Worker 已预加载 Seed 42。FastAPI 的 `/api/health` 提供服务状态，Worker 默认监听 `127.0.0.1:8001`。
- `npm --prefix apps/web run build` 通过。Vite 报告主包约 655 kB 的体积提示。浏览器验收覆盖真实推理、滚动取回已完成的结果、两套效率排行表、Bootstrap 分类，以及训练收敛的全部 51 个单模型组合。
- 本地运行 `python -m pytest -q tests/unit/test_export_web_data.py tests/unit/test_inference.py`，8 项通过。在仅含 Git 跟踪文件的副本中复测，2 项通过、6 项按预期跳过。

## 下一轮文档与仓库数据

1. 从 [README](../../README.md)、[文档索引](../INDEX.md) 和 [实验数据盘点](../experiment-data-inventory.md) 开始对齐项目入口。数据盘点里的“前端与推理入口”一行仍描述早期空前端与 E1 演示，需要替换为本次冻结的三页展示、FastAPI 和 Qwen 启动方式。
2. 用 [Batch A 逐 Run 索引](batch-a/model-index.json)、[Encoder 冻结清单](../../reports/batch-b-final/2605f6766085/freeze.json)、[Qwen 正式测试验收](../../reports/batch-b-s4/formal/test-acceptance.json) 核对报告数字、Run ID 和链接。训练与验证补充实验、探索性比较各自标注范围。
3. 盘点 Git 内的报告、配置、汇总 JSON 与本地 `data/`、`artifacts/` 证据。原始样本、逐样本预测、训练日志、权重和模型缓存通过本地归档交接。优先确认索引与指纹完整，再决定打包范围。
4. 课程报告版式位于独立工作树 `.worktree/report-style`，分支 `codex/report-style-v1`。整理报告内容时以正式结果为取数入口，保留该工作树的模板与编译验收记录。
