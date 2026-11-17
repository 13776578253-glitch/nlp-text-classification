# nlp-text-classification 当前状态

| 模块 | 已有内容 | 待验证或实现 |
| --- | --- | --- |
| 数据 | GoEmotions 官方下载器、TSV 解析、28 标签和分区 ID 校验 | 下载正式数据，记录 SHA-256 和标签分布 |
| E1 | TF-IDF + OvR LinearSVC，验证集阈值选择，保存与重载，独立最终测试命令 | 全量训练、验证与推理复验 |
| E2–E5 | TextCNN、BiLSTM、Transformer 与 Word2Vec 消融配置草案 | 训练代码、权重和正式结果 |
| 评估 | Micro-F1、Macro-F1、Samples-F1、Hamming Loss、Subset Accuracy 和逐标签指标 | 基于正式实验做类别分布与错误分析 |
| API / Web | FastAPI 推理接口与 React 演示原型 | 安装前端依赖后做浏览器端到端验收 |
| 工程与报告 | 单元测试、合成数据训练测试、CI、LaTeX 报告骨架 | 汇总真实 Run，生成报告表格和图表 |

下一阶段按顺序运行 `nlp-text-classification download`、`nlp-text-classification validate` 和 `nlp-text-classification train-e1`，检查验证集结果并用训练产物复验推理。模型与阈值选定后保留该 Run 供最终测试。E1 形成可复查的记录后，再进入共享 PyTorch 训练流程和 E2。
