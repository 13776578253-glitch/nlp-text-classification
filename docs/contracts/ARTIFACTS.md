# Batch A Run 与证据契约

原始实验留在 `artifacts/` 的 Git 忽略目录。计划、状态与契约位于 `docs/`。版本控制的验收记录引用本地原始证据的相对路径、Run ID 和哈希。

## 单次尝试

每个运行尝试排他创建 `artifacts/runs/<dataset_id>/<experiment_id>/<run_id>/`。记录基线、HPO、受控补跑、复验或重试用途，关联批次、数据版本、配置和种子。

| 文件 | 内容 |
| --- | --- |
| `run_manifest.json`、`resolved_config.yaml` | Run/批次/Study ID、Trial 编号、实际命令、代码 SHA、环境与设备、数据与配置指纹、参数、队列用途、起止时间 |
| `events.jsonl`、`stdout.log`、`stderr.log` | 真正的训练、早停、剪枝、故障或续跑事件与未经覆盖的输出 |
| `history.csv` | 每 epoch loss、验证指标、学习率、梯度检查、耗时和训练更新次数。SVM 记录一次拟合。 |
| `checkpoint/` | 获选模型，标签、预处理和词表指纹，以及中断恢复所需的模型、优化器、调度、epoch 和 RNG 检查点 |
| `thresholds.json`、`metrics_validation.json`、`predictions_validation.jsonl` | 分数类型、标签顺序、候选阈值及选择记录。逐行保存样本 ID、真实标签、全精度分数及离散预测。 |
| `quality.json`、`result.json` | 质量检查报告及输入/重算证据，PASS、FAIL、PRUNED 或 INCOMPLETE、退出码、耗时、停止原因和恢复来源 |
| `metrics_test.json`、`predictions_test.jsonl` | 进入正式冻结清单的 Run 才有此输出，与冻结版本和评估器绑定 |

## 批次与 Optuna

`artifacts/suites/<suite_execution_id>/` 保存队列状态、单写者 Optuna SQLite 库、Trial 导出、候选槽映射、复验名单、`PERFORMANCE_REVIEW.json`、审计记录与 `final_runs.yaml`。独立 E1 验证继续使用旧的平铺 Run 格式。

- 每个逻辑 Trial 持久化 Study、候选槽、参数与范围版本、验证结果、耗时、停止原因、状态和对应 Run ID。Study 以数据、模型和配置版本标识，串行访问。队列与数据库重启时先核对版本并对账已完成 Trial。原参数和预算计数在恢复前后保留。
- 对过期 RUNNING 状态核对活跃锁和最后日志。尝试中断后标记 INCOMPLETE。恢复可用的 checkpoint 时新建 Run，保存 `resume_from` 与状态指纹；失败重试保存 `retry_of`。同参数重试占重试槽，沿用原逻辑候选槽，记录各尝试实际耗时。终止后的原日志与原配置保持可读。
- 审计清单保存 14 个主 Run 和进入终测的复验 Run。冻结后对结果使用独占状态和原子发布，保留中断留痕，并通过样本 ID、标签顺序、分数、阈值和脚本版本重算表格与统计。
- 批次输出实时命令、当前 Run ID、训练阶段或 epoch、已完成项、验证指标和日志路径。批次 `progress.md` 随关键事件更新，完整输出与逐轮历史保留在追加日志。Codex 按阶段、首批训练完成、可恢复异常与验收门槛同步实际进度。

[Optuna RDBStorage 与串行 SQLite 约束](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html)
