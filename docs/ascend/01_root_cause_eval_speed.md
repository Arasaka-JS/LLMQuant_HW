# 评测为什么"dequant 缓存路径"快很多 —— 根因分析

> 目标模型：`checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507`（48 层 / 128 专家 / top-8）
> 量化产物：`LiftQuant/qmodels/Qwen3-30B-A3B-Instruct-2507/*.pth`
> 环境：`skw_quant_env`（torch 2.8.0+cu128 / transformers 5.9.0），本机加速器为 A800-SXM4-80GB（SM80, cc 8.0）

## 本次规划

- 回答一个具体现象：同一份量化权重，`e2e_utils` 里"有 `*.dequant.pth` 缓存"的层评测明显更快。
- 只做**只读**测量与代码定位，给出根因与证据，不改代码（修复另见 `02_materialize_dtype_fix.md`）。
- 顺带记录一个数据事故：`-layer0.dequant.pth` 是过期缓存。

## 结论（先说结果）

"dequant 缓存快" 不是因为它更省算力，而是因为它**绕过了两个隐藏开销**：

1. **FWTLinear 的 `_weight_fp` 是 float32**（不是 bf16），权重常驻显存翻倍；
2. **`FWTLinear.forward` 每次调用都把整块权重从 fp32 转一次激活 dtype**（`.to(x)` 是真拷贝），并且 MoE 下要**按命中专家重复支付**。

缓存路径把量化层换成了**原生 `nn.Linear`**，权重在 `model.to(target_dtype)` 时**只转换一次**，之后每次前向就是一次纯 bf16 matmul。

## 代码事实

### 事实 1：`uint8 - 0.5` 的 dtype 提升让解包结果变成 fp32

`LiftQuant/quantize/tmplinear.py:457-479` `FWTLinear.get_weight()`：

```python
if self.packed_flag:
    weight = (self.unpack_bits_uint8(self.packed_weight).reshape(self.oc, -1) - self.maxq / 2) * self.scale
```

`unpack_bits_uint8()`（同文件 `495-504`）返回 `uint8`；实测提升规则：

```
uint8 - 0.5            -> torch.float32
(uint8 - 0.5) * bf16   -> torch.float32
```

因此 `get_weight()` 的输出 dtype **与模块 dtype 无关，恒为 fp32**（在 fp16 模块与 bf16 模块上都实测过）。

### 事实 2：`forward` 每次调用都重铸整块权重

`LiftQuant/quantize/tmplinear.py:521-526`：

```python
def forward(self, x):
    weight_fp = getattr(self, '_weight_fp', None)
    if weight_fp is not None:
        return F.linear(x, weight_fp.to(x), self.bias)   # 每次前向都 fp32 -> bf16
```

`.to(x)` 在 dtype 不同时是**真拷贝**：每调用一次读 4B/元素、写 2B/元素。这是 **O(权重)** 的固定开销，与 token 数无关。

### 事实 3：MoE 让这个开销按"命中专家数"重复

`LiftQuant/models/qwen3_moe_per_expert.py:38-63` 的 `PerExpertQwen3MoeSparseMoeBlock.forward`：

```python
hit_experts = expert_mask.sum(dim=(-1, -2)).nonzero().flatten()   # 设备->主机同步
for expert_idx in hit_experts:
    expert_idx = int(expert_idx)                                  # 又一次同步
    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])    # 又一次同步
    current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[...]
```

- 30B-A3B 每层 128 专家 × 3 投影 = **384 个 `FWTLinear`**；
- 每个命中专家都要各付一次"全权重重铸"（哪怕它只分到 1 个 token）；
- prefill 2048 token 时命中全部 128 专家 → 每层 384 次重铸 + 512 次 Python 模块调用。

## 实测证据（真实 layer-0 权重）

### 1) dequant 缓存本身全是 fp32

```
qmodels/Qwen3-30B-A3B-Instruct-2507/Qwen3-30B-A3B-Instruct-2507+24to8-layer0.dequant.pth
dtype 直方图: [('torch.float32', 388)]      # 388 个张量
total: 2491.4 MB
```

它由 `LiftQuant/e2e_utils.py:352` 的 `module._weight_fp.detach().cpu()` 写出，因此直接证明了 `_weight_fp` 的 dtype。

### 2) 单模块 / 整层 MoE 的前向差距

| 对象 | plain bf16 `nn.Linear` | FWTLinear（fp32 缓存，现状） | FWTLinear（bf16 缓存，修好后） |
|---|---|---|---|
| 单模块 768×2048 (T=32) | 0.0233 ms | 0.0360 ms (**1.44×**) | 0.0231 ms (0.99×) |
| 整层 MoE，T=1 | 10.26 ms | 14.25 ms (**1.39×**) | 10.49 ms (1.02×) |
| 整层 MoE，T=32 | 11.40 ms | 15.36 ms (**1.35×**) | 11.66 ms (1.02×) |
| 整层 MoE，T=128 | 11.61 ms | 16.03 ms (**1.38×**) | 12.04 ms (1.04×) |

倍数**与 token 数无关** → 说明瓶颈是"固定权重流量"而非算力。

### 3) 主机同步代价（逐专家循环的另一半开销）

```
.nonzero() 每次同步            : 0.042 ms/层   -> 48 层约 2.0 ms/forward
torch.where/int() 逐专家同步    : 0.133 ms/层   -> 48 层约 6.3 ms/forward
```

### 4) 显存

`_weight_fp` fp32 让 layer0 专家部分常驻 **2416 MB**，bf16 只需 **1208 MB**。

## 事故：`-layer0.dequant.pth` 是过期缓存（已处理）

```
layer0: pth=09-18 13:51  dequant=09-10 15:50  max|diff|=0.0309  mean|diff|=0.0039  mean|ref|=0.0177  bitequal=False
layer1: pth=09-09 13:42  dequant=09-10 15:51  bitequal=True
layer2: pth=09-09 14:07  dequant=09-10 15:51  bitequal=True
```

`-layer0.pth` 在 9/18 13:51 被重新量化，而缓存是 9/10 的旧产物。`LiftQuant/e2e_utils.py:227-231` 只判断"文件是否存在"，会**静默使用旧权重**。
处置：已把该文件改名为 `Qwen3-30B-A3B-Instruct-2507+24to8-layer0.dequant.pth.stale-0910`（未删除，可恢复），下一次评测会自动用新 `-layer0.pth` 重建并写出**新的 bf16 缓存（约 1.2 GB）**。

## 遗留

- `main.py` 自己的 `evaluate()`（`run_quant.sh` 的 `--eval_ppl` 路径）不调用 `materialize()`，仍是"每次前向重新解包 + fp32 重铸"，行为与修复前一致。
- /tmp 内的过程日志会被清理；A/B 指标以仓库内 `lm_eval/eval_results/*.json` 为准。
