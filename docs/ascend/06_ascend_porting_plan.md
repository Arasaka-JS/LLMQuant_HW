# 昇腾 910 落地计划（bmm 结论、待验证清单、移植分期）

> 目标后端：Ascend 910（torch_npu + CANN）
> 当前状态：本机无昇腾卡（`npu-smi` 不存在、无 `torch_npu`），因此**本文的昇腾部分全部是"待真机验证"**。
> 已做的前置工作：device 抽象层（`04`）、MoE 向量化（`03`）、算子探测脚本（`05`）。

## 本次规划

- 回答"昇腾支持 bmm 吗"并给出源码级依据；
- 列出昇腾与 CUDA 的差异点（会影响现有实现的地方）；
- 给出 Phase D（真机探测 → 定实现）与 Phase E（整库移植）的分期与验证口径；
- 明确**在拿到卡之前不做**的事（纪律）。

## 结论：`torch.bmm` 支持，`torch._grouped_mm` 不支持

### `bmm` 支持的依据

`torch_npu` 的 `__init__.py`（v2.1.0 / v2.4.0 / master 均同）里：

```python
import torch_npu.op_plugin          # ← 把 ATen 算子映射到 NPU 实现的那一层
...
for name in dir(torch.ops.npu):     # ← NPU 专属算子被动态注册并暴露
    globals()[name] = getattr(torch.ops.npu, name)
    setattr(torch, name, ...)
```

`aten::bmm` 属于 op_plugin 覆盖的基础算子（映射到 CANN 的 BatchMatMul/MatMul），因此 `torch.bmm(a, b)` 在 device 为 `npu` 时可直接用，**代码无需改形状逻辑**。
另：`torch_npu` 也注册了 inductor 的 device op overrides（`_inductor_register_device_op_overrides`），所以 `torch.compile` 在昇腾上有基础支持。

### `torch._grouped_mm` 不可用

这是 CUDA 专属（且要求 SM90），本机实测报错：

```
RuntimeError: torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0
```

→ 我们选择的 **`torch.bmm` 路线在昇腾上也是正确路线**（不能用 grouped GEMM 替代）。

## 昇腾与 CUDA 的差异清单（影响现有实现）

| 差异点 | 说明 | 对当前实现的影响 | 处置 |
|---|---|---|---|
| **连续 vs 转置 B** | 昇腾偏好连续、format 友好输入（ND/NZ），转置视图可能触发格式转换 | 现在 `bmm(x_sorted, W.transpose(1,2))` 用的是 strided 视图 | 若探测显示 strided 明显更慢 → 在转换时**预存 bmm-ready 连续布局**（`(E,H,2I)`/`(E,I,H)`），运行时零开销 |
| **batch 并行度** | bmm 的 batch 维要切到 AI Core 上；decode 时每段只有 1 行，小 GEMM 效率低 | decode 可能仍需按命中专家 gather 后用更宽的 GEMM | 用探测脚本的 `decode` 形状实测再定 |
| **`uint8` 支持** | 昇腾对 `uint8` 的支持历来偏弱，而打包权重解包（`unpack_bits_uint8`）全靠 uint8 位运算 | 只影响**打包格式**的读取路径 | 探测里已含 `uint8_bitwise_and` / `uint8_to_float`；若不可用需转 int16/int32；也把"方案二：去掉打包"的价值抬高 |
| **`.item()` 同步** | 昇腾同步更贵，且在 graph 模式下禁止 | MoE 快路径每层 1 次（精确桶） | 已有可选 sync-free 桶（`MOE_SYNC_FREE_BUDGET>0`，建议 4096 ≈ M≤32） |
| **bf16** | 910B 支持 bf16；更早型号弱/不支持 | 决定 `_weight_fp` 与激活 dtype | 探测里有 `bf16_supported` 断言 |
| **版本矩阵** | 必须用 torch_npu 配套的 torch + CANN | 本机 `torch 2.8.0+cu128` 不能复用 | 在目标机按官方矩阵装；`device_utils` 已做可读报错 |
| **device 字符串** | `cuda:i` → `npu:i`；可见设备变量不同 | `e2e_utils` 的手工 `device_map` 用整数索引，靠 `device_count()` | `04` 已完成；启动脚本已支持 `ASCEND_RT_VISIBLE_DEVICES` |
| **集合通信** | HCCL 替代 NCCL（DDP 路径） | 影响量化训练（`stage_training.py` 的 DDP） | Phase E 处理 |

### 可能更优的路：昇腾原生 MoE 融合算子（**未确认**）

按昇腾生态（MindSpeed / MindIE）的 MoE 主路径，通常用一组融合算子：

```
npu_moe_gating_top_k(_softmax) / npu_moe_init_routing / npu_moe_compute_expert_tokens
npu_grouped_matmul / npu_moe_finalize_routing
```

如果目标机的 `torch_npu` 暴露了这些，把整个 MoE 前向压成 2–3 个融合大算子通常优于 bmm。
**但我没有从 torch_npu 源码确认这些名字**（算子是动态注册的），所以：**待真机探测**（`ascend_op_probe.py` 的第 3 段会枚举）。bmm 实现作为可回退方案保留。

## Phase D：真机探测 → 定实现（拿到卡后第一件事）

```
1. python allq/tools/ascend_op_probe.py --device npu --json /tmp/probe_910.json
2. 读三段输出：
   - 哪些算子 FAILED / >1ms（疑似 CPU 回退）        → 决定是否需要替换/降级
   - bmm: strided/contiguous 比值、bf16 支持         → 决定是否预存连续布局、dtype
   - torch.ops.npu 的 moe/grouped/matmul 候选        → 决定是否走融合算子
3. 最小端到端冒烟（建议补一个脚本）：
   加载 layer0 量化权重 → 一次 forward → 打印数值指纹 + 耗时（fast_moe on/off）
4. 用 device_utils 切后端：EVAL_DEVICE=npu:0 跑一遍 30B layer0 lm-eval，
   与 CUDA 的指标（hellaswag 0.625 / piqa 0.875 / winogrande 0.75）对比
```

判定口径：指标一致 + 单层前向耗时 ≤ CUDA 量级；若 strided-B 或 uint8 有瓶颈 → 按上表处置。

## Phase E：整库 `cuda → npu` 移植（量化训练线，成本最高）

现状：`grep` 显示 **14 个文件**直接依赖 CUDA（`LiftQuant/models` + `quantize` 内就有 32 处 `cuda` 引用）：

```
chat/chat_bitblas_compile.py  chat/chat_quant_bitblas.py  e2e_utils.py  e2efinetune.py
gptq/gptq.py  lattice_generator.py  lattice_generator2.py  models/LMClass.py
quantize/tmplinear.py  quantize/stage_training.py  quantize/liftq.py
eval_gsm8k.py  main.py  quant_training_ddp_validate.py
```

重点项：

1. **`bitblas`**（仅 `chat/*`）：CUDA-only，必须放弃或改走 `moe_fast_eval` 这套 PyTorch 原生路径（与本次 MoE 改造方向一致）。
2. **`torch.linalg.svd` / `inv`**（`FWTLinear.find_nearest_fast`、`lattice_generator*.py`）：只在**量化训练**时需要；昇腾上可能不支持或极慢 → 需实测，或退回到 `fast=True` 的随机码路径 / 在 CPU 上算。
3. **`torch.amp.autocast(device_type='cuda')`**（`main.py`）：改用 `device_utils.autocast()`。
4. **DDP / 集合通信**：`stage_training.py` 的 DDP 与 `torch.distributed` 初始化需按 HCCL 调整（`init_training_ddp_if_needed`）。
5. **`uint8` 位运算**：`unpack_bits_uint8`（`tmplinear.py:495-504`）与 `pack_to_int8`（`505-519`）。
6. **手工 `device_map` / `dispatch_model`**：整数索引在 accelerate 下会跟随当前加速器，`device_count()` 已抽象；但 `'cpu'` 与 dtype 转换路径需真机确认。

## 拿到卡之前不做的事（纪律）

1. 不把 `npu_moe_*` / `torch_npu.*` 写进**默认路径**；只做"探测到才启用、否则回退"。
2. 不做 `cuda` → `npu` 的**硬编码替换**（会让现有 A800 直接跑不了）；只做抽象。
3. 不提交无法验证的"昇腾性能数字"；所有昇腾结论标注为待验证。
4. 每次改动都必须先在 CPU/CUDA 侧证明"行为不变"（见 `05` 的三层验证）。
