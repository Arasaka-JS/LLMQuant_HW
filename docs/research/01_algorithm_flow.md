# LiftQuant 算法流程

## 本次规划
- 从 `main.py -> liftq.py -> tmplinear.py` 梳理一次完整权重量化流程。
- 聚焦默认主线：加载 FP 模型、收集校准样本、逐层替换和校正、转换保存量化权重。
- 暂不展开端到端微调和 BitBLAS 推理实现。

## 1. 入口与参数
主入口是 `LiftQuant/main.py`。它读取模型路径、量化位宽、`expc`、校准数据、训练轮数、保存目录等参数，然后构造 `LMClass` 加载 Hugging Face CausalLM。

关键参数：
- `--model`：FP Hugging Face 模型路径。
- `--wbits`：权重量化位宽。
- `--expc`：维度提升/投影配置，如 `20to8`、`24to8`。
- `--calib_dataset`：校准数据，代码支持 `wikitext2` 和 `redpajama`。
- `--training_trans`：使用可训练变换矩阵，而不是固定 Hadamard 变换。
- `--finetuning_weights`：开启 Stage2 权重细调。

## 2. 校准输入捕获
`liftq()` 首先关闭 `model.config.use_cache`，把 embedding、norm 和第一层移到目标设备。随后用内部 `Catcher` 包装第 0 层，在前向传播刚进入第一层时截获 hidden states、attention mask 和 position embeddings。

这样做的目的不是训练整个模型，而是为逐层校正准备固定的层输入：
- `fp_outs`：当前 FP 层的输入/输出参考。
- `quant_inps`：量化模型当前层接收到的输入。
- `attention_mask`、`position_embeddings`：保证层前向和原模型一致。

## 3. 逐层量化
`liftq.py` 对每一层循环处理。每层先根据当前 `fp_outs` 计算 FP 层输出，再把目标线性层替换为 `TmpLinear`。

被替换的模块包括注意力和 MLP 的主线性层：
- Llama/Qwen 常见路径：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`up_proj`、`gate_proj`、`down_proj`。
- 代码也保留了 `in_proj_qkv`、`in_proj_z`、`out_proj` 等结构分支。

## 4. Stage0：缩放初始化
替换为 `TmpLinear` 后，代码会根据激活分布初始化 `a1`。直觉上，`a1` 会放大扰动更敏感或尺度更大的输入通道，使后续变换域里的权重更适合低比特表示。

实现上，`get_act_means()` 收集若干模块的激活统计，然后使用通道标准差比值初始化 `a1`，并把范围裁剪到大约 `[1, 16]`。

## 5. Stage1：训练变换与尺度
Stage1 的目标是让量化后的当前层输出逼近 FP 当前层输出。训练对象主要包括：
- `quantizer.alpha`：调节量化 scale。
- `a1`、`a2`：输入/变换域缩放。
- `Trans.linear*`：可训练变换矩阵参数。
- 可选原始权重：由 `transmask` 和相关学习率控制。

每个 batch 中，`TmpLinear.quant_tmpweight()` 先构造临时量化权重，再运行层前向，用 MSE 对齐 `fp_outs`。

## 6. Stage2：可选权重细调
如果开启 `--finetuning_weights`，代码会把 `TmpLinear` 分组转换为 `FWTLinear`，然后继续用 MSE 细调量化权重、scale、变换矩阵和缩放参数。

Stage2 的分组顺序大致覆盖：
- QKV 投影。
- 输出投影。
- MLP 上投影/门控投影。
- MLP 下投影。
- 全部模块。

默认 README 命令启用了 `--finetuning_weights`，因此主实验通常包含这个块内校正步骤。

## 7. 输出传播与保存
每层完成后，代码会用量化层产生新的 `quant_inps`，作为下一层的量化输入。`--align` 会周期性把 `quant_inps` 对齐回 `fp_outs`，控制误差累积。

最后：
- `FWTLinear.pack_to_int8()` 把低比特权重打包成 `uint8` buffer。
- `main.py` 可把完整模型权重保存到 `save_dir/net/net+expc.pth`。
- `--save_per_layer` 可按层保存，适合超大模型降低内存压力。
