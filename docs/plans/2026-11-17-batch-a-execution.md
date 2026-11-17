# Batch A V2 执行计划 · 修订 3

2026-11-17 · 适用仓库 `nlp-text-classification`

上一轮迁移提交为 `ac0253e refactor(batch-a): migrate to dual-dataset experiment layout`。双数据 v1 已完成实物审计和派生冻结。当前进度见 [ACTIVE](../current-task/ACTIVE.md)，后续入口见 [NEXT](../NEXT.md)。

本轮修订保留 Python 包、CLI、双数据集、14 个主结果位及文档分层。在 [批次配置](../../configs/experiment_suites/batch_a.yaml) 和契约里同步 HPO、复验、质量验收与终测规则。Optuna、正式训练器和冻结评估器随各阶段实现。

## 一项总任务，六阶段依次执行

| 阶段 | 工作顺序 | 通过条件 |
| --- | --- | --- |
| Stage 0 · 基线审计与冻结 | 复核 Git、中文原始数据与 Word2Vec、GoEmotions 官方数据、划分哈希及中文语义标签。生成派生分区与排除清单。按 train 分布复核长度。锁定矩阵、评价口径、搜索空间、预算和复验选择规则。 | 双数据 manifest、映射、审计证据与配置可重放。验证后创建 Stage 0 基线提交。 |
| Stage 1 · 基础设施 | 接入严格 YAML 解析、14 项展开、训练数据访问限制、任务适配器、指标、序列化、Run recorder。实现队列状态、Optuna 持久 Study、checkpoint 续跑和测试门禁接口。锁定依赖版本并核验 CUDA。 | 失效配置可定位，两个任务的最小训练和预测重算通过。模拟异常、中断恢复、预算计数及无 test 路径运行。 |
| Stage 2 · 七种配置冒烟 | 对两个任务实现 E1/E2/E3/E5。覆盖稀有标签、短句、padding 与 OOV。专项测试验证单层 RNN/LSTM/BiLSTM 的 Dropout、padding 不改变有效表示、BiLSTM 正反向末端隐藏状态拼接。检查梯度、重载、冻结 Embedding 并做 CUDA 预检。 | 14 个组合均通过拟合、验证、保存、重载和质量断言，循环专项测试全部通过。 |
| Stage 3 · 基础、HPO、复验 | 先跑 14 个 seed 42 基础配置，随后运行 Optuna；选验证最优配置并完成 E5 受控补跑。确认 14 个主候选后，按规则执行三 seed 复验和可选补充。 | 14 个主 Run 和 30 个必选复验 Run 均可追溯并通过质量验收。可选失败明示。任一主结果位或必选复验仍失败时记录阶段未通过和原始证据。 |
| Stage 4 · 独立审计、冻结与正式测试 | 独立复核者从验证原始预测重算指标，核对数据、配置、checkpoint、阈值和入选过程。一次签出完整冻结清单，然后正式评估其全部条目。 | 两套 test 的主表和已冻结复验产物齐全。模型和统计对照名单都来自冻结前记录。恢复只补未成功项目，成功输出保留。 |
| Stage 5 · 分析与交付 | 从冻结预测重算表格与区间，产出探索和最终配置表、稳定性、训练曲线及收敛、E5 消融、类别错误分析、质量验收、复现手册和 LaTeX 课程报告初稿。 | 结果可追到原始预测及脚本版本，重跑汇总可复现。缺失和失败按实记录。 |

每阶段通过后创建一条独立 Conventional Commit。保留现有历史和远端，阶段失败记录写到状态和本地原始证据。可恢复问题在规则与预算内修复、重试并继续，成功项从证据重用。

## Stage 0 数据依据

- 中文源自 `选题4 数据.zip` 内的 `Dataset.rar`，已按原始字节导入 `data/raw/movie_reviews_zh/`。本地审计 JSON 为 `data/processed/movie_reviews_zh/dataset_audit.json`，记录四个源文件 SHA-256 和分区统计。
- Train / validation / test 有效样本为 19,998 / 5,629 / 369，空行 0 / 2 / 3，原始 0/1 数量分别为 9,999/9,999、2,817/2,812、182/187。Train/validation 有 2 种重合文本，对应 2 条训练行，标签一致。Train/test、validation/test 的空白归一化完全重合均为 0。建议派生 train 排除跨分区的 2 行与同标签的额外 2 条集内重复，预期 19,994 条，评估分区保留原始划分。
- 中文人工复核再次确认原始 `0=正向、1=负向`，已写入 v1。二分类目标的 1 表示语义正向。
- 中文训练长度 P50/P90/P95/P99 为 43/59/62/68，原训练集仅 1 条超过 256。Word2Vec 文件头为 426,677 词、50d，向量 SHA-256 为 `6a3dc817b85269ea15c18fab1b023456bb70f1531eefeac1e15c9a6f38453071`。Stage 1–2 复核读入、词数和词表覆盖率。
- GoEmotions 官方源为 43,410 / 5,426 / 5,427 条，保留 28 个标签与原分区。派生 v1 为 43,211 / 5,416 / 5,427 条。英文 50d Word2Vec 只用派生 train 无监督训练。 [数据实测与排除记录](../stage0-data-audit.md)

现有可运行的 Stage 0 入口如下。正式批次命令在实现并验证后进入复现手册。

```powershell
.venv/Scripts/python.exe -m nlp_text_classification.cli download
.venv/Scripts/python.exe -m nlp_text_classification.cli audit-data --dataset goemotions
.venv/Scripts/python.exe -m nlp_text_classification.cli audit-data --dataset movie_reviews_zh
```

## Stage 3 选择顺序与预算

1. 14 个组合先跑各自的基础配置，固定 seed 42，记录基线和原始验证预测。
2. 每数据集为 E1 追加 16 个 Optuna 候选。为 E2、E3-RNN、E3-LSTM、E3-BiLSTM 各追加 6 个候选。预算、具体范围、剪枝与失败计数见 [实验契约](../contracts/EXPERIMENTS.md)。
3. 按统一 validation Macro-F1 选取可验收候选。E5 两组沿用同数据集 E2 入选的非 Embedding 参数，同 seed 共用其余初始化，最多补跑 4 个 seed 42 Run。完全匹配的 E5 基础 Run 可引用。独立 E5 HPO 本轮分配 0 Trial。
4. 每数据集复验 E1、E2、E5-TUNE、E5-FROZEN 和验证表现最高的一个 E3。固定入选超参数，以 42、123、2026 三个种子独立重新训练，共 30 个。每数据集最多再加一个三次训练的可选区块，依据方差或验证差距。见 [复验矩阵](../contracts/EXPERIMENT_MATRIX.md)。
5. 全程用 [质量验收标准](../contracts/EXPERIMENTS.md#自动训练验收) 决定能否入选。异常先诊断，预算内单次恢复或重试。可审计的正常剪枝留在 Trial 证据中。

| 训练来源 | CPU（E1） | GPU（神经模型） |
| --- | ---: | ---: |
| 两数据集 seed 42 基础配置 | 2 | 12 |
| HPO 初始候选 | 32 | 48 |
| E5 受控补跑 | 0 | 0–4 |
| 必选复验 | 6 | 24 |
| 常规计划合计 | 40 | **84–88** |
| 可选复验 | 两设备共享最多 6 次，按入选模型分配 | 两设备共享最多 6 次 |
| HPO 补充 | 0 | 0–16 |
| PERFORMANCE_REVIEW | 两设备共享最多 2 次 | 两设备共享最多 2 次 |
| 重试预留 | 4 | 8 |
| 单设备上限 | 52 | **120** |

算上可选、重试、保守剪枝补充和性能诊断，两设备总计最多 164 次尝试，含剪枝启动项。Stage 2 冒烟和 Stage 4 推理单独计时。常规 GPU 任务若每项平均 10 分钟，约 14–14.7 小时；若平均 30 分钟，约 42–44 小时。实际耗时根据真实训练更新。

现机已识别 RTX 4070 Ti SUPER、16,376 MiB、驱动 610.88。迁移时的虚拟环境可运行 Python 3.12.13、既有 E1 和测试，PyTorch/CUDA、PyYAML、Gensim 与 Optuna 的版本组合待 Stage 1 验证。资源重点是单卡长队列、外部词向量加载、逐 epoch 验证开销及 checkpoint/日志磁盘占用。先用预检与第一批真实 Run 估算峰值显存、单 Run 时长、每 Run 容量和总磁盘需求。OOM 处理与断点留档按契约执行。

## Stage 4 审计与 Stage 5 汇总

- 独立审计以可回放的验证证据为输入，检查 14 个主结果位与全部终测复验的质量、数据哈希、配置、权重、最佳 epoch、阈值及选取记录。冻结 `final_runs.yaml` 后一次调度正式评估，预计 44–50 个 checkpoint，其中 14 个主 Run、30 个必选复验和最多 6 个可选复验。
- 所有验证或测试分数都由全精度逐样本输出复算。主表报告独立 seed 42 主 Run；复验表列三个种子、均值、样本标准差和样本数。使用冻结前固定的中文配对、10,000 次有放回重采样与 95% 区间。GoEmotions 附逐标签 support、F1、稀有标签的可用区间及类别不平衡分析。 [评估契约](../contracts/EVALUATION.md)
- 训练记录、失败日志和 HPO Study 保存在本地 `artifacts/`。验收报告引用路径、Run ID 和指纹。LaTeX 初稿在 Stage 5 从正式数据生成。E4、RQ3、QLoRA 和复杂前端均在 Batch A 验收之外。 [产物契约](../contracts/ARTIFACTS.md)

## 相对原迁移计划的变化

| 处理 | 内容 |
| --- | --- |
| 替换 | GoEmotions 主选择从 Micro-F1 改为 Macro-F1，中文也以 Macro-F1 为主。Stage 3 从固定首轮训练扩展为基础配置、HPO、E5 受控补跑、跨 seed 复验。终测冻结清单新增入选的复验 Run。 |
| 保留 | Python 包和 CLI、两套数据结构、14 个主结果位、先独立审计后终测、E2/E5 受控对照、GPU 串行及 Stage 0–5 路线。 |
| 推迟 | 独立 E5 优化、E4 Batch B、复杂前端以及公开仓库发布决策。LaTeX 随实际结果在 Stage 5 生成。 |
