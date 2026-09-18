# 验证手册：没有昇腾卡时怎么验、有什么验不了

> 背景：目标后端是昇腾 910，但当前服务器**没有**昇腾卡（`npu-smi` 不存在、无 `torch_npu`）。
> 本文把"改动 → 怎么验 → 需要什么硬件"讲清楚，并提供可复现命令与证据索引。

## 本次规划

- 建立**三层验证模型**：哪些能证明、哪些只能"承诺"。
- 给出三个自检工具的使用方法与覆盖范围。
- 记录 A/B 协议（保证两次运行路径可比）与证据落点。

## 三层验证模型

| 层 | 能验什么 | 需要什么 | 为什么可信 |
|---|---|---|---|
| **L1 正确性** | 向量化 MoE 的数值/边界/结构替换；后端选择逻辑 | 仅 CPU（无需加速卡） | 正确性与设备无关：GEMM 行间独立、padding 行不回读；实测 fp32 相对误差 ~1e-6 |
| **L2 性能与集成** | 单层/整模型性能、显存、lm-eval 指标、加载链路 | 本机 CUDA（A800） | 与目标环境同代码路径 |
| **L3 后端特性** | 昇腾算子可用性/耗时、格式开销、融合算子、HCCL | **昇腾真机** | 无法在本机推断，必须实测 |

**纪律**：L3 相关代码一律做成"探测到才启用、否则回退"，不进默认路径；不在没有真机时写死 `npu_*` 调用，也不做无法验证的性能承诺。

## 工具 1：`allq/tools/moe_fast_selfcheck.py`（L1，23/23）

```bash
python allq/tools/moe_fast_selfcheck.py                 # 纯 CPU，秒级
python allq/tools/moe_fast_selfcheck.py --device cuda    # 也可在 GPU 上跑
python allq/tools/moe_fast_selfcheck.py --experts 64 --hidden 512 --inter 2048
```

覆盖：

- **dtype**：float32（紧容差 1e-5，证明算法等价）与 bfloat16（容差 1e-2）；
- **形状**：decode M=1、short M=3、prefill M=64、batch (2,16)；
- **病态路由**：全部 token 只走专家 0（其余专家 0 token）、每专家恰好 1 个 slot；
- **两种桶**：sync-free vs exact 必须一致，且用 `duplicate` 路由（**有放回**，故意违反 topk 互斥）验证 `assume_distinct_topk=False` 的安全上界；
- **guard-rail 回退**：把 `max_padded_rows=1` 强制走 `_forward_expert_loop()`，与参考循环比对；
- **结构替换**：`convert_moe_blocks_to_fast()` 替换块数、**旧 `_weight_fp` 无泄漏**。

参考实现就是文档里的逐专家循环，逐元素比对。

## 工具 2：`allq/tools/device_utils_selfcheck.py`（L1，18/18）

```bash
python allq/tools/device_utils_selfcheck.py
```

用 **stub 的 `torch_npu` / `torch.npu`** 覆盖 NPU 分支：后端判定、`device_count`、`resolve('auto')→'npu:0'`、拒绝 `cuda:0`、`synchronize` 派发、不可用时回退、以及两条"可读报错"断言。
**边界**：只验证选择逻辑，不验证真实昇腾算子。

## 工具 3：`allq/tools/ascend_op_probe.py`（L3，到真机跑）

```bash
# 真机（昇腾）
python allq/tools/ascend_op_probe.py --device npu --json /tmp/probe_910.json
# 对照（本机 CUDA，已跑过）
python allq/tools/ascend_op_probe.py --device cuda
```

输出三段：

1. **算子探测**（median ms）：`bmm_contig_B` / `bmm_strided_B` / `matmul_2d` / `argsort(stable)` / `bincount` / `cumsum` / `repeat_interleave` / `index_select` / `advanced_index_assign` / `gather` / `unique` / `topk` / `index_add` / `silu` / `uint8_bitwise_and` / `uint8_to_float` / `item_sync`；失败或 >1ms 的会被标出（>1ms 通常意味着**回退到 CPU**，那会带来隐式同步）。
2. **真实 MoE 形状下的 bmm**（prefill E=128,T=128 / decode E=8,T=1）：contiguous B vs **strided B** 的耗时比（决定是否预存 bmm-ready 连续布局）、bf16 支持。
3. **`torch.ops.npu` 里的融合算子候选**（关键字 `moe/grouped/matmul/topk/routing/expert/bmm`）→ 判断能否走昇腾原生 MoE 融合算子。

本机 CUDA 参考结果（A800）：所有算子 `ok`；`bmm_contig 0.264ms / bmm_strided 0.272ms (1.03×)`；prefill 形状 `gate_up 0.636ms / down 0.352ms`；bf16 支持。

## Fast MoE 的 A/B 协议（L2）

**关键点**：`e2e_utils` 只要发现 `{prefix}-layer{i}.dequant.pth` 就会走"纯 `nn.Linear` 缓存路径"，**根本不会构建 MoE 块**。所以 A/B 两次运行必须都从"重建"开始。

```bash
# 1) 用临时前缀只暴露 layer0 一个量化文件（其余层保持 FP），保证两次路径完全一致
mkdir -p /tmp/moe_ab
ln -sf <QMODEL_DIR>/Qwen3-30B-A3B-Instruct-2507+24to8-layer0.pth /tmp/moe_ab/ab30b-layer0.pth

# 2) 两次运行之间删掉缓存（第一轮会重新生成）
run() { CUDA_DEVICE=3 EVAL_BACKEND=liftquant \
        FP_MODEL_PATH=<...>/Qwen3-30B-A3B-Instruct-2507 \
        QUANT_MODEL_PATH=/tmp/moe_ab/ab30b LOAD_PER_LAYER=1 \
        TASKS=hellaswag,piqa,winogrande LIMIT=8 EVAL_DTYPE=bfloat16 EVAL_BS=8 \
        LIFTQUANT_WBITS=2 LIFTQUANT_EXPC=24to8 FAST_MOE=$1 \
        ./allq/scripts/eval_by_lmeval.sh; }
rm -f /tmp/moe_ab/ab30b-layer0.dequant.pth; run 0
rm -f /tmp/moe_ab/ab30b-layer0.dequant.pth; run 1
```

验收口径：

- 指标逐项一致（本次：hellaswag 0.625 / piqa 0.875 / winogrande 0.750）；
- 日志出现/不出现 `[fast_moe] layer 0: fused 1 MoE block(s)`；
- 结果 JSON 的 `run.fast_moe` 字段正确记录。

其他回归：

- **非 MoE 路径**：Qwen3-4B（dense，全量化）`FAST_MOE=1` 应复现 `acc=0.250 / acc_norm=0.625`，且**不得出现** `[fast_moe]` 行。
- **仅单层性能**：用真实 layer-0 权重直接对 `FastMoEExperts` 与逐专家循环做 A/B（见 `03` 的表）。

## 无卡也能防回归的护栏（建议保留）

`*.dequant.pth`、golden 输出等大文件不入库；推荐做法是**每次改动都重跑参考实现比对**（`moe_fast_selfcheck.py` 就是这个角色），而不是依赖存下来的 golden 张量——因为参考循环本身很短，重算比维护 golden 更可靠。

## 证据索引

仓库内（持久）：

- `lm_eval/eval_results/ab30b_liftquant_20260918T084149Z.json`（`fast_moe=False`）
- `lm_eval/eval_results/ab30b_liftquant_20260918T084619Z.json`（`fast_moe=True`）
- 4B 的 `..._liftquant_*.json`（`fast_moe=True` 冒烟）

过程日志（`/tmp`，会被清理，正文已内联关键数字）：

```
/tmp/phaseA.log /tmp/phaseA_fp32.log /tmp/phaseA_final.log   # MoE A/B、fp32 等价性
/tmp/phase0_gpu.log                                          # 修后基线 + sync-free 对比
/tmp/v2_moe.log /tmp/v2_smoke.log /tmp/v3_lmeval2.log        # 早期 smoke 与 lm-eval
/tmp/phaseC_lmeval.log /tmp/phase1_4b.log                    # device 接线 / 4B 回归
/tmp/ab30b_fast0.log /tmp/ab30b_fast1.log /tmp/ab30b_timing.txt
/tmp/phase2_forward.log /tmp/phase2_forward_fast.log         # 单次前向 on/off
/tmp/probe_cuda.json                                         # 算子探测（CUDA 基线）
```

数据产物：`LiftQuant/qmodels/Qwen3-30B-A3B-Instruct-2507/*.dequant.pth`、`...-layer0.dequant.pth.stale-0910`（过期缓存，已改名保留）。
