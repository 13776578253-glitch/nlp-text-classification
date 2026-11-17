# Batch A Run 产物契约

正式 Run 写入 `artifacts/runs/<dataset_id>/<experiment_id>/<run_id>/`。分配目录时即建立 manifest，记录 Run/批次 ID、启动命令、代码 SHA 与 dirty 状态、设备与软件版本、seed、数据和配置指纹、状态和时间。并发保留目录须使用排他创建。

| 文件 | 内容 |
| --- | --- |
| `resolved_config.yaml` | 运行时解析后的数据、特征、模型、阈值选择与设备参数 |
| `events.jsonl`、`stdout.log`、`stderr.log` | 阶段、epoch、警告、异常和原始日志 |
| `history.csv` | 神经模型逐 epoch loss、验证指标、学习率、耗时。E1 记录一次 fit 的时间与指标。 |
| `thresholds.json`、`metrics_validation.json`、`predictions_validation.jsonl` | 验证阈值证据、汇总/逐类指标和逐样本分数 |
| `checkpoint/` | 最佳模型、预处理组件、标签映射和可核查指纹 |
| `result.json` | 退出码、完成检查项、PASS/FAIL/INCOMPLETE，失败原因及恢复来源 |
| `metrics_test.json`、`predictions_test.jsonl` | Stage 4 冻结后发布的正式测试输出 |

批次队列状态与 `final_runs.yaml` 归入 `artifacts/suites/<suite_execution_id>/`。旧 E1 独立验证及推理继续读取平铺的本地旧 Run，独立入口生成的 Run 记为 `standalone_validation`。正式产物引用由批次管理器统一生成。
