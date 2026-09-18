# 向量化 MoE 前向（排序分桶 + 融合 bmm）

> 目标模型：Qwen3-30B-A3B-Instruct-2507（48 层 / 128 专家 / top-8 / hidden 2048 / moe_inter 768）
> 新增代码：`LiftQuant/models/moe_fast_eval.py`
> 环境：`skw_quant_env`（torch 2.8.0+cu128），A800-SXM4-80GB（SM80, cc 8.0）

## 本次规划

- 把 MoE 的"逐专家 Python 循环"换成 GPU 内向量化实现，且**只在评测/推理路径生效**（量化训练路径不动）。
- 记录设计决策的依据（实测数字），以及为正确性/健壮性加的守卫。
- 记录与 `e2e_utils` / lm-eval 的集成方式与端到端 A/B 结果。

## 现状：MoE 前向是怎么实现的

### 原生 transformers 5.9.0

`Qwen3MoeExperts.forward`（`transformers/models/qwen3_moe/modeling_qwen3_moe.py:215-250`）权重是**融合 3D 布局** `gate_up_proj (E,2I,H)` / `down_proj (E,H,I)`，但 forward 仍是：

```python
expert_mask = one_hot(top_k_index, E).permute(2,1,0)
expert_hit  = (expert_mask.sum(dim=(-1,-2)) > 0).nonzero()      # 设备->主机同步
for expert_idx in expert_hit:
    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])  # 又一批同步
    gate, up = F.linear(state, self.gate_up_proj[e]).chunk(2,-1)
    y = F.linear(act(gate)*up, self.down_proj[e]) * weight
    final.index_add_(0, token_idx, y)
```

（`modeling_qwen2_moe.py:295-331` 完全相同，Qwen3 只是继承。）

### LiftQuant 侧

- `LiftQuant/models/qwen3_moe_per_expert.py:12-63`：把 block 猴子补丁回**逐专家 `ModuleList[Qwen3MoeMLP]`**（为了逐专家 ckpt 原样加载 + 量化分支按专家处理）。forward 同样是循环 + `hit_experts = ...nonzero()`。
- 量化侧 `build_moe_grouped_fwt_eval`（`quantize/tmplinear.py:1123+`）把每个专家的 3 个投影换成 `FWTLinear`（组间共享 `Trans/a1/a2`）。
- 评测侧 `materialize()` 后每个 `FWTLinear` 就是一次 `F.linear`。

**调用链（每层每次前向）**：`Qwen3MoeDecoderLayer.mlp` → block.forward → `nonzero()`(同步) → `for e: int(e)`(同步) + `torch.where`(同步) → `Qwen3MoeMLP.forward` → **3× FWTLinear.forward** → `index_add_`。
prefill 命中 128 专家时 ≈ **512 次 Python 模块调用 + 128 次以上主机同步**；decode（top-8）≈ 24 次调用 + 9 次同步。

## 瓶颈实测（真实 layer-0 权重，bf16）

| 场景 | 现状（逐专家循环） | 向量化 | 理论下限 |
|---|---|---|---|
| prefill 每层（每专家 128 token） | **10.19 ms** | **1.07 ms** | ~0.8 ms（155 GFLOP@200T / 805MB@2TB/s） |
| decode 每层（1 token, top-8） | **0.857 ms** → 41 ms/token(48层) | **0.276 ms** → 13 ms/token | ~0.04 ms（只读 8 专家 75MB） |
| 主机同步 | `.nonzero()` 0.042 + `int()/where` 0.133 ms/层 ≈ **8.4 ms/forward(48层)** | 0 | — |

结论：**launch + 主机同步主导**（~26 µs/模块调用），真正的 matmul 只有 µs 级。

`torch._grouped_mm`（更理想的分组 GEMM）在 A800 上实测报
`RuntimeError: torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0`
→ 只能走 `torch.bmm`。

## 设计

`FastMoEExperts.forward(hidden_states (M,H), top_k_index (M,k), top_k_weights (M,k))`，全程 GPU、无主机循环：

1. `flat_e = top_k_index.reshape(-1)`、`slot_tok = arange(M).repeat_interleave(k)`
2. `order = argsort(flat_e, stable=True)` → `sel_e`、`tok`、`slot_weights`（按专家分组，组内保持原顺序 ⇒ 可复现）
3. `counts = bincount(sel_e, minlength=E)`、`starts = cumsum(counts)-counts`、`pos = arange(S) - starts.repeat_interleave(counts)`
4. `tmax = _select_tmax(...)`，分配 `x_sorted (E, tmax, H)` 并 `x_sorted[sel_e, pos] = hidden.index_select(0, tok)`
5. **一次** `torch.bmm(x_sorted, gate_up.transpose(1,2))` 同时算出 gate 与 up（gate 在前半、up 在后半，与 HF `Qwen3MoeExperts` 的布局一致）
6. `act(gate) * up` 后**再一次** `torch.bmm(..., down.transpose(1,2))`
7. `down[sel_e, pos]` 取回真实行 → 乘 routing 权重 → `index_add_` 回输出

每层约 8~10 次 kernel launch（vs 现状 384 / 24），零主机同步。
padding 行是零且**永不回读**（用 `(sel_e,pos)` 取回，`pos` 恒在 `counts[e]` 内），因此不影响结果（GEMM 行间独立）。

## 结构转换与内存

`convert_moe_block_to_fast(block)`（`moe_fast_eval.py:381+`）：

- `stack_experts_from_block()` 把每个专家的 `gate_proj/up_proj` 拼成 `(E,2I,H)`、`down_proj` 堆成 `(E,H,I)`（**预分配 + 逐专家填充**，见"踩坑"一节）；
- 同时支持已量化的 `FWTLinear`（读 `_weight_fp`，缺失则先 `materialize()`）与未量化的 `nn.Linear`（部分量化场景）；
- 转换后**释放**旧 per-expert 的 `_weight_fp`/`packed_weight`/`scale`，显存只保留一份：layer0 专家部分 **1208 MB**（`leaked_fp_cache=0`，自检断言）；
- `convert_moe_blocks_to_fast(module)` 递归替换（支持 MoE 块嵌在任意层级），返回被替换的块名列表。

## 关键决策（都有实测依据）

### 1) 分桶策略默认走「精确桶」，sync-free 只作为可选开关

`_select_tmax()`（`moe_fast_eval.py:143+`）有两种桶：

- **精确桶**：`tmax = max(counts)` —— 需要 1 次 `.item()` 主机同步（实测 0.042 ms/层），但 padding 最少；
- **sync-free 桶**：数据无关的上界，零同步，但要把所有专家 padding 到上界。

上界有两种，由 `assume_distinct_topk` 决定：

| 上界 | 前提 | 大小 | 说明 |
|---|---|---|---|
| `num_slots = M*k` | 无前提（一个专家可能独占所有 slot） | 大 | 安全但浪费 |
| `num_tokens = M` | top-k 返回**互斥**专家（`torch.topk` 语义） | 小 k 倍 | 本仓库真实 router 满足 |

实测（同一份 layer0 权重，`assume_distinct_topk=True`）：

| case | 精确桶 | sync-free | 提速 | 是否逐位相同 |
|---|---|---|---|---|
| decode M=1 | 1.124 ms | 1.104 ms | 1.02× | **True**（bound=1，与精确桶同值） |
| decode M=8 | 1.190 ms | 1.189 ms | 1.00× | False（bf16 分块差异） |
| prefill 512 / 2048 | 1.801 / 4.374 ms | 同（被硬上限挡回精确桶） | 1.00× | False |

结论：**默认 `SYNC_FREE_ROW_BUDGET = 0`（恒精确桶）**；sync-free 只在"很短的序列"上划算，且 M=128 时 padding 会让单层时间接近翻倍，所以还必须配硬上限。

### 2) 三道守卫（防止病态输入把评测打崩）

| 常量 | 默认 | 作用 |
|---|---|---|
| `SYNC_FREE_ROW_BUDGET` | `0` | >0 才启用 sync-free 桶；`bound * num_experts <= budget` |
| `SYNC_FREE_MAX_SLOTS` | `256` | 无论 budget 多大，`bound <= 256` 才允许 sync-free（否则 prefill 会按 `tmax=S` 分配 ~8.6 GB / 慢 22×） |
| `MAX_PADDED_ROWS` | `1<<16` | `E*tmax` 超限则切到 `_forward_expert_loop()`（基于 fused 权重的逐专家回退，正确但慢），避免病态路由 OOM |
| `ASSUME_DISTINCT_TOPK` | `True` | 声明 router 语义；设 `False` 则退回 `num_slots` 安全上界 |

`ASSUME_DISTINCT_TOPK` 的依据：`Qwen3MoeTopKRouter.forward` 用 `torch.topk(probs, top_k)`，topk 每个 token 返回**互不相同**的专家 ⇒ `counts[e] <= num_tokens`。
若假设不成立，失败模式是**响亮的** `IndexError`（`x_sorted[sel_e, pos]` 越界），不是静默错值。

## 实测（真实 layer-0 权重，bf16，A800）

### 逐专家循环 vs 向量化

| case | OLD | NEW | 提速 | max\|diff\|（bf16） |
|---|---|---|---|---|
| decode(1,1) | 2.01 ms | **1.14 ms** | 1.77× | 3.9e-03 |
| decode(1,8) | 10.45 ms | **1.19 ms** | **8.78×** | 3.1e-02 |
| prefill(1,512) | 31.40 ms | **1.80 ms** | **17.42×** | 6.3e-02 |
| prefill(1,2048) | 32.84 ms | **4.38 ms** | **7.50×** | 3.1e-02 |
| batch(2,64) | 31.52 ms | **1.38 ms** | **22.82×** | 3.1e-02 |

`max|diff|` 是 bf16 输出层面的差异（bf16 的 eps≈7.8e-3）；**算法等价性**由 fp32 A/B 证明：

```
fp32: decode 2.4e-07 / short 9.5e-07 / prefill512 4.8e-06 / prefill2048 1.1e-05   (max|d|)
      相对误差 3.3e-07 ~ 1.2e-06  -> 仅为浮点累加顺序差异
```

## 集成（评测路径）

`LiftQuant/e2e_utils.py`：

```python
def load_quantized_model(..., fast_moe=True, moe_sync_free_row_budget=MOE_SYNC_FREE_BUDGET):
```

- 位置：**`materialize()` 之后、写 `*.dequant.pth` 之后、`dispatch_model` 之前**；
  - 在写缓存之后：缓存读的是 `_weight_fp`，转换会释放它；
  - 在 dispatch 之前：fused buffer 会随 `dispatch_model` 一起搬到目标设备（在 dispatch 之后转换会 OOM，见"踩坑"）。
- 命中时打印：`[fast_moe] layer 0: fused 1 MoE block(s)`。
- `allq/eval/eval_quant_lmeval.py`：`--fast-moe/--no-fast-moe`（默认 on 便于默认提速，A/B 用 `--no-fast-moe`）、`--moe-sync-free-row-budget`，两者都写入结果 JSON。
- `allq/scripts/eval_by_lmeval.sh`：`FAST_MOE=1`（默认）/`MOE_SYNC_FREE_BUDGET=0`，并在表头打印实际取值。

## 端到端 A/B（30B layer0 + 4B 回归）

为保证两次运行**路径完全可比**，用临时前缀只暴露 layer0 一个量化文件，并在两次运行之间删掉 dequant 缓存：

```bash
mkdir -p /tmp/moe_ab
ln -sf <...>+24to8-layer0.pth /tmp/moe_ab/ab30b-layer0.pth     # 只有 layer0 会被替换
FAST_MOE=0/1  LOAD_PER_LAYER=1  QUANT_MODEL_PATH=/tmp/moe_ab/ab30b ...
```

| 验证 | 结果 |
|---|---|
| 指标（30B layer0，LIMIT=8） | `FAST_MOE=0` 与 `1` **完全一致**：hellaswag 0.625 / piqa 0.875 / winogrande 0.750（JSON：`ab30b_liftquant_20260918T084149Z.json`、`...084619Z.json`） |
| 墙钟（30B layer0） | 277 s → 270 s（抖动级；只量化 1/48 层且加载占主导） |
| 单次前向（同模型内实测） | prefill512 1083.8 → **1021.8 ms (1.06×)**；prefill2048 1254.9 → **1235.0 ms (1.02×)**；decode 253.9 → 256.5 ms/tok（0.99×，噪声） |
| 4B（dense，无 MoE）回归 | `FAST_MOE=1` → hellaswag acc 0.250 / acc_norm 0.625，**与改动前一致**；日志中无 `[fast_moe]` 行（非 MoE 路径零影响） |

**解读**：单层 MoE 收益是 **7.5–22.8×**，端到端只体现"量化层占比"。当前只量化 1/48 层 ⇒ 端到端 2–6%（prefill）/ 无感（decode），属预期。

## 踩坑与修正

1. **`torch.stack` 的 2× 峰值**：先收集 list 再 `stack` 会让"逐专家源张量 + fused 副本"同时存活，在近满显存/大 MoE 上 OOM（实测在 dispatch 之后转换时触发）。已改为**预分配 + 逐专家填充**（`moe_fast_eval.py:346-356`）。
2. **在 dispatch 之后转换会 OOM**：30B 已在 GPU 上占 ~72 GB，再要 ~2.4 GB 瞬时峰值即失败 → 结论：转换必须放在 dispatch 之前（现已如此）。
3. **`assume_distinct_topk` 引入后自检失败**：自检原先用 `torch.randint`（**有放回**）生成路由，同一 token 可能重复同一专家 ⇒ 违反 `counts[e] <= M` ⇒ `IndexError`。已修自检为 topk（互斥）并补 `duplicate` 模式对照。详见 `05_verification_playbook.md`。

## 遗留

- **全量量化（48 层）的端到端收益与显存**未测：bf16 materialize ≈ 58 GB，需要 4×80 GB 或另设计存储策略；这是吃满 MoE 收益的前提。
- 进一步优化方向（未做）：按 count 分桶降低 padding 浪费（实测 pad-to-max 浪费 ~23%）、`torch.compile`/CUDA graph（去掉主机同步后才可行）、昇腾原生融合算子（见 `06`）。
