# nlp-text-classification 实施路线

| 阶段 | 核心交付物 | 退出条件 |
|---|---|---|
| Stage 0 | Monorepo、需求、架构、数据协议、CI、合成数据冒烟 | `pytest` 全通过，Git 初始提交 |
| Stage 1（下一步） | GoEmotions 下载与校验；E1 完整训练与验证 | 记录数据 SHA-256、类别分布和验证指标，重载模型后可预测 |
| Stage 2 | PyTorch 训练器、TextCNN (E2) | 测试模型形状、梯度、Checkpoint、真实训练 |
| Stage 3 | BiLSTM (E3)、Embedding 消融 (E5) | 与 E2 数据和指标一致；冻结 Embedding 自动检验 |
| Stage 4 | DistilBERT、ModernBERT-base（E4） | 主结果可复现；ModernBERT-large 视精力增强 |
| Stage 5 | API/Web 全模型、报告生成、20NG 轻量基准 | 五组实验最终测试、图表可追溯、展示稳定、文档完整 |

## 三人协作建议

- 工作流 A：数据协议、E1、E2、E5 和 Word2Vec 消融；
- 工作流 B：E3、E4、深度学习复现实验；
- 工作流 C：评估规范、FastAPI、Web、LaTeX 和测试；
- 合并所有分支前：`pytest`、结构与结果契约审查。

不要以训练硬件极限或大型模型参数作为 Stage 4/5 的阻塞条件。
