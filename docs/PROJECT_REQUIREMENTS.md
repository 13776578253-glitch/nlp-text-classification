# nlp-text-classification 项目需求（v1）

## 1. 课程原题

选题 4「自然语言处理之文本分类」：建立能够对输入文本进行分类的系统，在给定数据集或网络公开数据集上进行测试和效果对比。优先深度学习，涵盖词嵌入、神经网络训练；原题列举 Word2Vec、CNN、LSTM 等作为技术选项。公开数据允许替换，不强制使用 20 Newsgroups。此处为课程原题的概要，原始资料由用户提供。

## 2. 本项目最终选择

- 研究主题：细粒度情绪多标签识别。
- 主数据：GoEmotions **official simplified**（英文 Reddit 评论、28 标签）；沿用官方 train/dev/test。**不把 58,009 条 raw corpus 与 simplified 拆分混为一谈。**
- 20 Newsgroups：末尾一轮轻量 TF-IDF + SVM 主题分类基准；单标签 20 分类与 GoEmotions 的结果分别呈现。
- 技术：Python、scikit-learn、PyTorch、Transformers、FastAPI、TypeScript、React、Vite、LaTeX。
- 团队：3 人；课程得分仍是重要验收目标，但将系统设计为可继续研究的个人项目。
- 可用主要计算资源：本地 RTX 4070 Ti SUPER 16GB、32GB 内存。E4 提升模型能力，不将硬件上限作为课程研究目标。

## 3. 两个研究问题

**RQ1.** TF-IDF/SVM、TextCNN、BiLSTM、Transformer 在 GoEmotions 多标签分类任务上的表现如何？稀有标签、共现标签、相近情绪的性能差异是什么？

**RQ2.** 更强的预训练 Encoder 在相同数据划分、统一评估协议下能带来多少收益？优先比较 DistilBERT 与 ModernBERT-base；ModernBERT-large 为增强实验。

另行讨论的 16GB 显存极限优化、QLoRA 和大语言模型微调吞吐研究，不纳入本仓库需求和验收，另开项目时独立处理。

## 4. 必须完成的实验

| ID | 方法 | 目的 | 最低验收 |
|---|---|---|---|
| E1 | TF-IDF + OvR LinearSVC | 统计模型基线 | 训练/验证/最终测试、模型保存和加载 |
| E2 | TextCNN + 随机可训练 Embedding | CNN 文本特征学习 | PyTorch 训练、最佳权重、指标及曲线 |
| E3 | BiLSTM + Embedding | 双向序列建模 | 与 E2 同拆分和指标，独立模块 |
| E4 | DistilBERT 微调；ModernBERT-base 对照 | 预训练 Encoder 对照 | 至少两个模型的公平实验；large 为加强项 |
| E5 | TextCNN：随机 / Word2Vec 微调 / Word2Vec 冻结 | 词嵌入消融 | 共享架构、只改变 Embedding 策略，检查冻结权重 |

**注意：** GoEmotions 是多标签，不再使用 single-label Accuracy 作为唯一或主要指标。课程报告核心为 Micro-F1、Macro-F1、按类 F1、Recall，以及阈值选择过程。

## 5. 非目标

- 不将 Emotion Wheel 中不同心理学分类体系直接视为 GoEmotions 的真实标签层级。
- 不将情绪分类输出包装成心理诊断、人格测量或用户心理健康结论。
- 不在首版实现分布式训练、线上持续训练、用户上传语料自动学习或大规模监控平台。
- 不以 E4 模型参数尽可能大为首要完成条件。
