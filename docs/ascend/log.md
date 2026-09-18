# 昇腾/推理性能专项 —— 开发日志（按时序）

> 记录范围：从"评测为什么慢"的问题提出，到 MoE 向量化、后端抽象、端到端 A/B 与文档沉淀的全过程。
> 环境：`skw_quant_env`（torch 2.8.0+cu128 / transformers 5.9.0），A800-SXM4-80GB（SM80），本机无昇腾卡。
> 约定：每一步记录「目标 / 动作 / 命令 / 关键输出 / 结论」；数值均来自本机实测。

## 本次规划

- 把开发过程（含被否决的方案与踩坑）完整留档，便于后续接昇腾时不必重新推导。
- 与时序无关的结论已拆分到 `01`–`06`；本文只保留"怎么一步步走到这里"。

## 起点：任务与约束

- 用户诉求演进：① 先理解 LiftQuant 训练+评测全流程与"打包/解包"环节；② 判断"显卡不支持 bitblas 后是否还需要打包"；③ 解释"dequant 缓存为什么快很多"；④ 让量化后的 MoE 前向更高效；⑤ 后续要在**昇腾 910** 上跑。
- 硬约束：改 auto-round 源码需先征得同意（本次**完全未碰** `auto-round/`）；仓库无 CI/测试，验证只能用定向 smoke + 自检脚本。

## 时间线

### ① 通读流程，厘清"打包/解包"到底有几处

**动作**：读 `main.py`、`quantize/liftq.py`、`quantize/tmplinear.py`（`FWTLinear`）、`e2e_utils.py`、`chat/chat_quant_bitblas.py`、`chat/chat_bitblas_compile.py`，并 grep `bitblas` / `pack|unpack`。

**关键输出**：

- 仓库里其实有**两处**打包：
  - A：`FWTLinear.pack_to_int8()`（`tmplinear.py:505-519`）——把 1-bit 码按 8 个/字节压进 `uint8`，由 `get_weight()` 解包。**这是 checkpoint 的落盘格式**；
  - B：`chat/chat_quant_bitblas.py` 的 `pack_weight_int1()`——把 A 的位序重排成 BitBLAS int1 布局，只有 M=1 decode 用真 kernel。
- `import bitblas` 全仓库仅出现在 `chat/chat_quant_bitblas.py:24`；`main.py` 的评测、`allq/eval/eval_quant_lmeval.py` 的 liftquant 后端都**不导入**它。
- 实测（真实 `-layer0.pth`）：`packed_weight` 解包后是纯 `{0,1}`，且 `|w| == scale/2` 恒成立；打包路径与"不打包的伪量化公式"**逐元素相同**（identical fraction = 1.000000，max diff = 0.0）；单模块体积 50.3 MB(fp16 展开码字) → 3.1 MB(packed) = **16×**。

**结论**：打包 A 不是为 bitblas 存在的，它是"3 bit/权重"能在磁盘上成立的前提，也是权重唯一载体；删掉它等于换 checkpoint 格式（会退回 16 bit/权重）。用户"不需要打包"的直觉只对 B 成立。

### ② 诊断"dequant 缓存为什么快很多"

**动作**：读 `e2e_utils.load_quantized_model` 的两条分支；检查 `-layer*.dequant.pth` 的 dtype/体积；在真实 layer0 上做 A/B。

**关键输出**：

```
dequant 缓存 dtype 直方图: [('torch.float32', 388)]   total 2491.4 MB
uint8 - 0.5            -> torch.float32
(uint8 - 0.5) * bf16   -> torch.float32
单模块 forward: plain-bf16 0.0233 ms | FWTLinear(现状) 0.0360 ms (1.44x)
整层 MoE T=32 : plain-bf16 11.40 ms | FWTLinear 15.36 ms (1.35x)
```

根因：`get_weight()` 因 dtype 提升恒返回 fp32；`forward` 里 `weight_fp.to(x)` **每次调用**重铸整块权重；MoE 下按命中专家重复支付。另外 `-layer0.dequant.pth` 是 **9/10 的过期缓存**（`layer1/2` bit-exact，`layer0` 不一致）。

**结论**：慢的根源是"缓存 dtype fp32 + 每次前向重铸"，**不是** bitblas 或打包的锅。详见 `01`。

### ③ 修复一：`materialize(dtype=)`（Act）

**动作**：改 `tmplinear.py` `FWTLinear.materialize`：`get_weight()[:, :self.ic]` 后一次性 `.to(scale.dtype)`；签名加 `dtype=None`（调用点无需改）。先做 monkeypatch 预验证再落盘。

**关键输出**：

```
VALUE PRESERVED == legacy.to(bf16) : True (torch.equal)
BIT-EXACT vs -layer1.dequant.pth   : True (max|diff|=0)
_weight_fp 6.29 MB -> 3.15 MB ；整层 MoE 2416 MB -> 1208 MB
单模块 forward 1.44x -> 0.99x ；整层 MoE 1.35-1.39x -> 0.97-1.04x
4B 全量化整模型: prefill512 53.9->49.5ms(1.09x) prefill2048 135.7->122.3ms(1.11x) decode 47.7->43.9ms/tok(1.09x)
4B lm-eval hellaswag：0.250 / 0.625（与修复前一致）
```

**结论**：数值零变化（只改"缓存用什么 dtype 存"），显存减半、每量化层前向回到原生 Linear 水平。详见 `02`。

### ④ 现状盘点：MoE 前向为什么低效

**动作**：读原生 `Qwen3MoeExperts.forward` 与 LiftQuant 的 `PerExpertQwen3MoeSparseMoeBlock.forward`；在真实 layer0 上测量并尝试 `torch._grouped_mm`。

**关键输出**：

```
逐专家循环每层 9.15~10.19 ms（每专家 128 token）
.nonzero() 0.042 ms/层 + int()/where 0.133 ms/层  ≈ 8.4 ms/forward（48 层）
torch._grouped_mm -> RuntimeError: only supported on devices with compute capability = 9.0
融合 gate+up 的分桶 bmm：1.07 ms/层（9.5x）
```

**结论**：瓶颈是 **launch + 主机同步**（~26 µs/模块调用），实际 matmul 只占 µs；方案定为"GPU 内排序分桶 + 融合 gate/up 的 2 次 bmm"。详见 `03`。

### ⑤ 方案实施（用户批准按 A → B → C 顺序，每步都在本机验完）

**Phase A：fast MoE 的真实权重 A/B**

- 新增 `LiftQuant/models/moe_fast_eval.py`（`FastMoEExperts` / `FastMoEBlock` / `stack_experts_from_block` / `convert_moe_block(s)_to_fast`）。
- fp32 严格等价性：decode 2.4e-07 / prefill2048 1.1e-05（相对 3e-7~1e-6）→ 算法等价，bf16 差异纯属分块舍入。
- 性能（真实 layer0）：decode(1,1) 1.77× / decode(1,8) 8.78× / prefill512 17.42× / prefill2048 7.50× / batch(2,64) 22.82×；结构替换后无 `_weight_fp` 泄漏，fused 1208 MB。
- 无卡自检：`allq/tools/moe_fast_selfcheck.py`（当时 17 项全过）。

**Phase B：`.item()` 处理 + 无卡自检**

- 先按"固定 bucket"思路实现，随后**用实测纠正了默认值**：
  - sync-free（bound=`num_slots`）在 CUDA 上 decode 是 0.97×/0.87×（更慢）；
  - 若 budget 设很大让 prefill 也走 sync-free → `tmax=S=16384` 会分配 8.6 GB、**慢 22×**（96 ms/层）。
- 于是：默认 `SYNC_FREE_ROW_BUDGET = 0`（恒精确桶，零回归），并加两道硬上限：`SYNC_FREE_MAX_SLOTS = 256`（sync-free 只在短序列生效）、`MAX_PADDED_ROWS = 1<<16`（超限切 `_forward_expert_loop` 回退，防病态路由 OOM）。
- 更新自检，覆盖 fp32/bf16 / 各种形状 / 单专家路由 / 每专家 1 slot / 两种桶 / 回退路径 / 结构替换与释放。

**Phase C：device 抽象层**

- 新增 `LiftQuant/device_utils.py`（懒加载 `torch_npu`、`resolve()` 返回字符串、可读报错）；接线 `e2e_utils`（`device_count()`/`empty_cache()`）、`--device auto`、`EVAL_DEVICE` + `ASCEND_RT_VISIBLE_DEVICES`。
- 新增 `allq/tools/device_utils_selfcheck.py`（stub NPU，18/18）。
- CUDA 行为不变证据：4B lm-eval 指标接线前后一致（0.250 / 0.625）。

### ⑥ 集成 + 端到端 A/B

- `e2e_utils.load_quantized_model(..., fast_moe=True, moe_sync_free_row_budget=0)`：在 **materialize + 写 dequant 缓存之后、dispatch 之前**转换，打印 `[fast_moe] layer i: fused N MoE block(s)`。
- `--fast-moe/--no-fast-moe`、`--moe-sync-free-row-budget`（并写入结果 JSON）；`eval_by_lmeval.sh` 增加 `FAST_MOE`/`MOE_SYNC_FREE_BUDGET`。
- 30B layer0 A/B（临时前缀 + 两次之间删缓存，保证路径可比）：指标**完全一致**（hellaswag 0.625 / piqa 0.875 / winogrande 0.750），墙钟 277 s → 270 s。
- 同模型内单次前向：prefill512 1083.8 → 1021.8 ms (1.06×)、prefill2048 1254.9 → 1235.0 ms (1.02×)、decode 253.9 → 256.5 ms/tok（噪声）。
- 4B（dense）回归：`FAST_MOE=1` 复现 0.250 / 0.625，且无 `[fast_moe]` 行。

### ⑦ 文档沉淀

- 本目录 `docs/ascend/`（`README` + `01`–`06` + 本日志）。素材来自各阶段日志与仓库内 `lm_eval/eval_results/*.json`；关键数字内联在正文（因为 `/tmp` 日志会被清理）。

## 用户做过/要求过的关键决策

| 决策 | 内容 |
|---|---|
| 打包/bitblas 判定 | 认可"bitblas 只影响 chat 路径"；`pack_to_int8` 保留为存储格式 |
| 修复顺序 | 要求先修 fp32 问题（已完成），再讨论方案二（去打包） |
| MoE 优化 | 选定"向量化 bmm"方向，并按 A→B→C 顺序推进、每步本机验证 |
| 昇腾前提 | 明确"当前无昇腾卡"，要求给出无卡验证方案（→ `05`） |
| 文档 | 要求把开发过程写入 `docs/ascend/`，选"8 个文件全量" |

## 踩坑与修正

1. **误以为 sync-free 一定更快**：实测 CUDA 上 padding 比 sync 贵（decode 0.87×）；且大 budget 在 prefill 上会灾难性变慢（22×）。→ 默认改回精确桶 + 硬上限。
2. **`torch.stack` 的 2× 峰值**：先收集 list 再 stack 会让源张量 + fused 副本同时存活 → 在 dispatch 之后转换时真实 OOM。→ 改为预分配 + 逐专家填充。
3. **在 dispatch 之后转换会 OOM**（30B 已在 GPU 上占 ~72 GB）→ 转换固定在 dispatch 之前。
4. **`torch.device("npu:0")` / `torch.amp.autocast("npu")` 在无 torch_npu 的 torch 构建里报错**（自检抓到）→ `device_utils` 捕获并重抛可读错误；`resolve()` 不依赖 `torch.device`。
5. **自检与真实 router 不变量不一致**：自检用 `torch.randint`（有放回）生成路由，违反 `counts[e] <= num_tokens`（topk 互斥）→ `IndexError`。→ 自检改用 topk 生成，并补 `duplicate` 模式验证安全上界。
6. **过期 dequant 缓存静默生效**：`e2e_utils` 只判断文件存在，`-layer0.dequant.pth` 是旧权重的产物 → 已改名 `.stale-0910` 保留。
7. **`/tmp` 里的长命令挂住终端**：后期改为 `nohup ... > log 2>&1 &` + 轮询日志，避免前台阻塞。

## 被否决 / 搁置的方向

| 方向 | 状态与理由 |
|---|---|
| 方案二：彻底去掉 `pack_to_int8`（存 fp16 `weight`） | 搁置。数值可行（已验证 pack→unpack 与伪量化公式逐位相同），但 checkpoint ×16；且需要连带改加载/转换/chat。 |
| chat 的 BitBLAS int1 kernel | 搁置。目标卡不支持 bitblas，且该路径只影响 chat；MoE 改造后可改走原生 PyTorch 路径。 |
| `torch._grouped_mm` | 否决（本机 A800 与昇腾均不可用）。 |
| CUDA graph / `torch.compile` | 推迟。现有逐专家循环含主机同步无法编译；MoE 向量化后已具备条件，但收益需实测。 |
| 按 count 分桶降 padding | 推迟。实测 pad-to-max 浪费仅 ~23%，收益有限。 |

## 遗留问题

1. **全量量化（48 层）的端到端收益与显存未测**（bf16 materialize ≈ 58 GB）。
2. **`main.py` 自带 `evaluate()`（`--eval_ppl`）未接 `materialize`**，仍是旧行为。
3. **昇腾真机验证**：`ascend_op_probe.py` 未在 910 上跑过；`npu_moe_*` 融合算子是否存在未确认（Phase D）。
4. **整库移植（Phase E）未开始**：14 个文件依赖 CUDA，含 SVD/inv、uint8 位运算、HCCL、autocast。
5. `run_moe_group_quant.sh` 的工作区改动非本次产生（开始前即 `M`）。
