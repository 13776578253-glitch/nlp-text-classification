# Stage 2 CPU 验收

Windows Python 3.12.13、PyTorch 2.9.1 CPU。两数据集各七个配置全部完成合成拟合、验证、保存与加载。日志位于 `artifacts/batch-a-smoke-cpu/progress.jsonl`，汇总为同目录的 `summary.json`。每个 E5-FROZEN 的初始、终止和获选 Embedding 指纹一致。

22 项 pytest 通过。循环网络的专项测试涵盖单层外部 Dropout、尾部填充内容对有效表示无影响、BiLSTM 拼接末端正向和起点反向状态。恢复测试模拟中断并从上一 epoch checkpoint 恢复，验证预测逐字节等于持续训练。

正式训练准入使用独立的 CUDA 冒烟汇总 `artifacts/batch-a-smoke/summary.json`。CPU 证据用于实现验证。
