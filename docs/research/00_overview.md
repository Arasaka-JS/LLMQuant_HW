# LiftQuant 学习总览

## 本次规划
- 建立从方法目标到源码入口的学习路线。
- 先覆盖权重量化主线，后续再追加激活/KV 量化、BitBLAS 推理、端到端微调等专题。
- 本文只回答“LiftQuant 在解决什么问题、用什么核心机制解决、应该按什么顺序读代码”。

## 问题定位
LiftQuant 是一个大模型后训练量化框架，目标是在 2-bit、2.5-bit、3-bit 等非标准位宽下压缩 LLM 权重，并通过块内校正降低精度损失。

传统均匀量化常把每个权重通道直接映射到有限整数格点；当位宽很低时，格点过稀，误差会迅速放大。LiftQuant 的核心思路是先把权重向量放到一个更适合量化的变换域，再用“维度提升/投影”的方式构造更丰富的低比特码字集合。

## 核心机制
- 维度提升与投影：用 `expc` 表示从高维二值码字到低维向量的映射，例如 README 中 3-bit 使用 `24to8`，2.5-bit 使用 `20to8`。
- 投影矩阵 `M`：预训练矩阵位于 `LiftQuant/lattice/`，可由 `LiftQuant/lattice_generator2.py` 生成；运行主实验时通常不需要重新生成。
- 块内校正：`LiftQuant/quantize/liftq.py` 按层收集校准输入，替换线性层为临时量化层，优化变换、缩放和量化尺度，使量化层输出贴近 FP 层输出。
- 部署形态：训练/校正阶段使用 `TmpLinear`，最终转换为 `FWTLinear` 并将量化权重打包到 `uint8`。

## 推荐阅读顺序
1. `LiftQuant/README.md`：先确认官方命令、模型推荐配置和部署建议。
2. `LiftQuant/main.py`：理解参数、模型加载、校准数据缓存、调用 `liftq()` 和评测逻辑。
3. `LiftQuant/quantize/liftq.py`：理解逐层量化、校准输入捕获、Stage1/Stage2 校正。
4. `LiftQuant/quantize/tmplinear.py`：理解 `TmpLinear`、`FWTLinear`、`bit_channel_convert()` 和打包逻辑。
5. `LiftQuant/lattice_generator2.py`：理解投影矩阵 `M` 如何被优化出来。

## 当前文档体系
- `01_algorithm_flow.md`：主量化算法流程。
- `02_lattice_projection.md`：维度提升、投影矩阵和码字搜索。
- `03_code_map.md`：源码入口和关键类函数导读。
