# Batch A 执行与进度

Windows PowerShell 下使用项目虚拟环境和无缓冲输出。GPU 预检及 14 项冒烟通过后进入正式训练。

```powershell
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-a smoke --device cuda --output artifacts/batch-a-smoke
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-a run --device cuda
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-a audit
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-a final-test --device cuda
.venv/Scripts/python.exe -u -m nlp_text_classification.cli batch-a report
```

训练终端逐阶段输出 Run ID、epoch 与 validation Macro-F1，长 epoch 约每 20 秒记录当前批次。正式任务的实时快照为 `artifacts/batch-a/progress.md`，增量事件为 `artifacts/batch-a/progress.jsonl`。冒烟或另设输出目录时，同名文件位于该目录。每轮详细数据在对应 Run 内的 `history.csv` 和 `stdout.log`。

```powershell
.venv/Scripts/python.exe -m nlp_text_classification.cli batch-a status
.venv/Scripts/python.exe -m nlp_text_classification.cli batch-a resources
Get-Content artifacts/batch-a/progress.jsonl -Tail 12 -Wait
```

中断后用同一 suite 和输出目录重跑原命令。队列沿用已完成的尝试，恢复记录关联原 Run。独立审计通过后发布冻结清单，正式测试重复调用时只补齐未成功的条目。

批次入口检查 Windows 的系统物理和提交内存，启动时两者均须余 8 GiB。运行中每秒复测，任一低于 6 GiB 时记录 `PAUSED_MEMORY`，中断当前 Run。Windows 执行进程的提交内存上限为 4 GiB，并使用低于普通的进程优先级。CPU 计算线程限制为 2。已写完的 checkpoint 和原日志随队列保留。

实现验收使用隔离的 CPU 合成数据。14 个配置的逐轮记录位于 `artifacts/batch-a-smoke-cpu/`。隔离的完整流程运行了基础训练、Optuna、复验、独立冻结、50 个合成测试和报告生成，日志为 `artifacts/batch-a-workflow-check.log`。
