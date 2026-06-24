# LiftQuant 源码导读

## 本次规划
- 给后续阅读代码提供稳定入口和调用关系。
- 标出容易误读的实现细节和环境依赖。
- 不列完整文件树，只记录影响理解的方法边界。

## 主入口
`LiftQuant/main.py` 负责参数解析、随机种子、模型加载、校准数据缓存、调用 `liftq()`、保存权重和评测。

关键调用链：
1. `LMClass(args)` 从 Hugging Face 路径加载 tokenizer/config/model。
2. `get_loaders()` 或 `get_redpajama()` 构造校准数据。
3. `liftq(lm, args, dataloader, logger)` 执行逐层量化。
4. `evaluate()` 可计算 PPL 或调用 `lm_eval` 做任务评测。

## 模型封装
`LiftQuant/models/LMClass.py` 是轻量封装，主要提供：
- `AutoConfig.from_pretrained()` 和 `AutoModelForCausalLM.from_pretrained()`。
- `device_map='cpu'` 初始加载，再由量化流程逐层搬到 CUDA。
- `use_fast=False` 的 tokenizer。

这意味着主流程设计成逐层搬运，避免一次性把所有层长期放在 GPU 上。

## 逐层量化核心
`LiftQuant/quantize/liftq.py` 是最重要文件。阅读时建议按下面顺序定位：
- `liftq()` 开头：关闭 cache、识别 Llama/Qwen 层结构、捕获第 0 层输入。
- norm fuse 部分：把 RMSNorm 权重融合进后续 Linear，减少量化路径中的独立缩放。
- `replace_linear_with_TmpLinear()`：把目标 `nn.Linear` 替换成可训练临时量化层。
- Stage1 训练循环：优化变换和缩放，使量化层输出对齐 FP 层输出。
- Stage2 分组细调：可选地转换成 `FWTLinear` 后继续细调。
- 层末更新 `quant_inps`：决定误差如何传递到下一层。

## 量化层实现
`LiftQuant/quantize/tmplinear.py` 包含两个核心类：
- `TmpLinear`：训练/校正阶段使用，保留原始 `nn.Linear`，通过 STE 量化构造临时权重。
- `FWTLinear`：最终部署形态，保存投影域权重、缩放、变换和打包后的 `packed_weight`。

关键函数：
- `TmpLinear.find_params()` 初始化量化尺度。
- `TmpLinear.quant_tmpweight()` 在训练循环中构造当前量化权重。
- `FWTLinear.bit_channel_convert()` 执行投影码字转换，是 LiftQuant 区别于普通均匀量化的关键实现。
- `FWTLinear.pack_to_int8()` 将量化值打包存储。

## 投影矩阵生成
`LiftQuant/lattice_generator2.py` 用优化方式生成 `./lattice/{D_in}to{D_out}.pt`。当前仓库已包含预训练矩阵，README 明确主结果通常可跳过生成。

注意：脚本默认维度是 `32to16`，不要误以为直接运行会生成 README 中的 `20to8` 或 `24to8`。

## 激活和 KV 量化入口
`LiftQuant/models/int_llama_layer.py` 定义 `QuantLlamaAttention` 和 `QuantLlamaMLP`，包含 activation quantizer、QK/PV matmul quantizer、bias/scale/rotation 开关等。当前主 README 命令把 `abits/kbits/vbits` 设为 16，重点仍是低比特权重量化。

## GPTQ 相关代码
`LiftQuant/gptq/` 保留 GPTQ 量化器和 Hessian 统计逻辑。`main.py` 中 `--act-order`、`--true-sequential`、`--percdamp` 等参数来自 GPTQ 传统流程，但 LiftQuant 主线还叠加了变换域、投影码字和块内校正。

## 容易踩坑
- Python 命令应从 `LiftQuant/` 目录运行，因为代码使用 `./cache`、`./lattice`、`../log` 等相对路径。
- `datautils.py` 和 `datautils_e2e.py` 有硬编码数据集缓存路径，换环境时需要检查。
- `e2efinetune.py` 依赖当前 checkout 中不存在的 `datautils_block`，运行前需要补齐或确认来源。
- 主量化、PPL、lm-eval 和 BitBLAS 聊天都依赖实际模型权重、CUDA 和数据集下载，不适合作为轻量单元测试。
