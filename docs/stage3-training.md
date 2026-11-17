# Stage 3 正式训练验收

| 项目 | 结果 |
| --- | --- |
| Seed 42 基础矩阵 | 14/14 通过 |
| Optuna | 80 个 Trial，74 个通过、6 个剪枝。每个神经模型 Study 的完整 Trial 均达到下限。 |
| 主 Run | 14/14 通过，来源为 7 个 HPO、4 个受控消融和 3 个基础 Run。 |
| 三种子独立复验 | 30/30 通过，种子为 42、123、2026。 |
| 追加复验 | 中文 BiLSTM 的 42、123、2026 三个 Run 均通过。 |
| 性能诊断 | 14 个主 Run 已检查。GoEmotions RNN 的额外诊断通过技术质量验收，验证 Macro-F1 为 0.24571。诊断记录单列。 |

队列共保存 134 次运行尝试，其中 126 次通过、6 次剪枝、2 次因内存压力处理期间中断。中断与续跑证据保留在原 Run 目录。

队列证据为 `artifacts/batch-a/queue.json`，SHA-256 为 `8c2e2a5bd6c51727b07157d224d6e07c94282803418eef8e6b612bbd066749e9`。性能诊断见同目录 `PERFORMANCE_REVIEW.json`，逐轮记录见 `terminal.log` 与各 Run 的 `history.csv`。正式测试由 Stage 4 独立审计放行。
