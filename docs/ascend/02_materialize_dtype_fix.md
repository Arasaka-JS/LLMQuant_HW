# 修复一：`FWTLinear.materialize()` 按目标 dtype 缓存权重

> 前置阅读：`01_root_cause_eval_speed.md`（问题定位）
> 环境：`skw_quant_env`（torch 2.8.0+cu128），A800-SXM4-80GB（SM80），真实 layer-0 量化权重

## 本次规划

- 修掉"`_weight_fp` 恒为 fp32 + 每次前向重铸"这个开销，且**不改变任何数值**。
- 给出可复现的证据（与旧行为逐位对比、与既有 `*.dequant.pth` 逐位对比、端到端指标不变）。
- 不做与 MoE 前向结构有关的改动（那块见 `03_moe_vectorized_forward.md`）。

## 补丁

文件：`LiftQuant/quantize/tmplinear.py`，`FWTLinear.materialize`（原 528-548 行）。

```python
@torch.no_grad()
def materialize(self, dtype=None):
    if hasattr(self, '_weight_fp'):
        return self
    if dtype is None:
        # After ``model.to(target_dtype)`` these already hold the evaluation dtype
        if 'scale' in self._parameters:
            dtype = self._parameters['scale'].dtype
        elif 'weight' in self._parameters:
            dtype = self._parameters['weight'].dtype
        else:
            dtype = torch.float32
    weight_fp = self.get_weight()[:, :self.ic]
    if weight_fp.dtype != dtype:
        weight_fp = weight_fp.to(dtype)          # 一次性转换
    self.register_buffer('_weight_fp', weight_fp.contiguous(), persistent=False)
    for name in ('packed_weight',):
        if name in self._buffers:
            del self._buffers[name]
    if 'scale' in self._parameters:
        del self._parameters['scale']
    return self
```

要点：
- **只改"缓存用什么 dtype 存"**；`get_weight()` / `forward()` / `pack_to_int8()` 一行未动，解包与旋转仍在 fp32 里算，最后只做一次 cast —— 与"加载 `*.dequant.pth` 缓存"这条路径完全同构。
- `dtype` 参数可选；`e2e_utils` 的三处调用（`345`/`358` 行）都在 `model.to(target_dtype)` 之后，`self.scale.dtype` 正好是评测 dtype，所以**调用点无需改动**。
- `forward` 里的 `weight_fp.to(x)` 从此成为 no-op（同 dtype/device）。

## 数值验证（这是"零风险"的依据）

### 1) 与旧行为逐位相同

```
legacy materialize dtype : torch.float32
patched materialize dtype: torch.bfloat16   (MB 6.29 -> 3.15)
VALUE PRESERVED == legacy.to(bf16) : True   (torch.equal)
```

### 2) 与既有 `*.dequant.pth` 逐位相同

用 `-layer1.pth` / `-layer2.pth`（其缓存晚于 ckpt，未过期）复现 `get_weight()` 并与缓存比对：

```
layer1: max|diff|=0.000000  bitequal=True
layer2: max|diff|=0.000000  bitequal=True
```

说明 `get_weight()` 的数学没有变，只是"存这份结果的 dtype"变了。

### 3) 端到端指标不变

- Qwen3-4B（全量化、252 个 `FWTLinear`）lm-eval hellaswag（LIMIT=8）：**acc=0.250 / acc_norm=0.625**，与修复前完全一致（`lm_eval/eval_results/`）。

## 收益（实测）

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| `_weight_fp` dtype（单模块） | float32 | bfloat16 | — |
| 单模块常驻 | 6.29 MB | **3.15 MB** | 减半 |
| layer0 专家部分常驻 | 2416 MB | **1208 MB** | 减半 |
| 4B 全量化（252 模块） | 14533 MB | **7267 MB** | 减半 |
| 单模块前向 | 0.0360 ms（1.44×） | **0.0231 ms（0.99×）** | 回到原生 Linear |
| 整层 MoE（128×3，T=32） | 15.85 ms（1.38×） | **11.89 ms（1.04×）** | ~1.33× |
| 4B 整模型 prefill 512 / 2048 | 53.9 / 135.7 ms | **49.5 / 122.3 ms** | 1.09× / 1.11× |
| 4B 整模型 decode | 47.7 ms/tok | **43.9 ms/tok** | 1.09× |
| `*.dequant.pth` 体积（layer0） | 2491 MB | **1246 MB** | 减半 |

> 注：4B 是 dense 模型，端到端 1.09–1.11× 是"全量化 + 每层都吃满"的上限；只量化少量层时端到端收益按层数摊薄（见 `03` 的端到端 A/B）。

## 集成路径的确认

30B layer0 走完整评测链路（`e2e_utils.load_quantized_model` → `materialize` → 写缓存）后产出的缓存：

```
/tmp/moe_ab/ab30b-layer0.dequant.pth   1 245 840 825 bytes (≈1.2 GB)
```

体积恰为旧 fp32 缓存（2491 MB）的一半 → 修复在集成路径生效。

## 回滚

补丁只影响 `materialize()` 的缓存 dtype；如需回到旧行为，把 `weight_fp = weight_fp.to(dtype)` 去掉即可（不建议：会同时失去速度与显存收益，且数值不变）。
