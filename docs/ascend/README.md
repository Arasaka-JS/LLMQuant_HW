# 昇腾 / 推理性能专项 —— 文档索引

> 本目录记录"为昇腾 910 做准备"这条线上的分析、改动、验证与遗留。
> **状态：未完成** —— 昇腾真机验证、全层量化的端到端、整库移植、`chat/` 迁移均未做，见「未完成清单」。
> 代码根目录：`LiftQuant/`；评测入口：`allq/eval/eval_quant_lmeval.py` + `allq/scripts/eval_by_lmeval.sh`。

## 本次规划

- 把"评测慢的根因 → 修复 → MoE 向量化 → 后端抽象 → 端到端 A/B → 昇腾落地计划"这条链完整留档。
- 明确区分**已验证**（CPU / 本机 CUDA 实测）、**待真机验证**（昇腾）、**已搁置**三类结论。
- 与既有文档分工：`docs/research/` 讲算法与源码导读，`docs/moe/` 讲 Qwen3-MoE 适配，本目录讲**推理性能与昇腾落地**。

## 未完成清单（明确未做完）

> 本次工作**只覆盖"评测/推理性能 + 昇腾准备"这条线**；下列事项**都还没做完**，请勿当作已完成。

### A. 完全没有验证（需要昇腾真机）

| 项 | 状态 |
|---|---|
| `ascend_op_probe.py` 在 910 上运行 | **未跑过**，目前只有 CUDA 基线 |
| `device_utils.device()`（返回 `torch.device`）的昇腾路径 | 仅用 stub 验证了逻辑；真机需 torch_npu 注册 `npu` 设备类型后才能确认 |
| 昇腾原生 MoE 融合算子（`npu_moe_*` / `npu_grouped_matmul` 等）是否存在 | **未确认**，仅为候选；靠探测脚本枚举 |
| 昇腾上的 bmm 实际耗时、strided-B 格式开销、`uint8` 位运算支持、bf16 支持 | **均未实测** |
| 昇腾上的端到端评测（`EVAL_DEVICE=npu:0`） | **未跑过** |

### B. 写好了但没接 / 没测

| 项 | 状态 |
|---|---|
| Phase D 的"最小端到端冒烟脚本"（真机一条命令出结论） | 只在 `05`/`06` 里描述了做法，**脚本尚未编写** |
| 整库 `cuda → npu` 移植（Phase E） | **未开始**；14 个文件依赖 CUDA（含 SVD/inv、`uint8`、HCCL、autocast） |
| `main.py` 自带 `evaluate()`（`--eval_ppl`）接入 `materialize` | **未做**，仍是"每次前向重新解包 + fp32 重铸"的旧行为 |
| `chat/`（BitBLAS chat 与 `packed_weight` 依赖） | **未动**；目标卡不支持 bitblas，需改走原生 PyTorch 路径才可用 |
| 全量量化（48 层）的端到端收益与显存 | **未测**；bf16 materialize ≈ 58 GB，需 4×80GB 或改存储策略 |

### C. 讨论过但主动搁置（不是漏做）

- **方案二：彻底去掉 `pack_to_int8`**（改存 fp16 `weight`）——数值已验证可行（pack→unpack 与伪量化公式逐位相同），但 checkpoint ×16 且需连带改加载/转换/chat，暂不做。
- `torch._grouped_mm`——本机（SM80）与昇腾均不可用，**已否决**。
- `torch.compile` / CUDA graph、按 count 分桶降 padding——**推迟**（收益未定）。

### D. 验证覆盖的边界（重要）

- 正确性是**容差等价**，**不是 bit-exact**：fp32 相对误差 ~1e-6；bf16 输出 max diff 可达 6e-2（分块 GEMM 累加顺序不同）。
- 端到端 A/B 只覆盖 **30B layer0（1/48 层）** 与 **4B dense 回归**；**全层量化的端到端未做**。
- 本目录所有昇腾相关结论均标注为**待真机验证**，不要直接当结论引用。

## 一页结论

1. **评测慢的根因**（见 `01`）：`FWTLinear._weight_fp` 因 `uint8 - 0.5` 的 dtype 提升恒为 **float32**，且 `forward` 每次调用都 `weight_fp.to(x)` 重铸整块权重；MoE 下按命中专家重复支付。修掉后**数值不变**（与旧 fp32 结果及既有缓存逐位相同）。
2. **收益**：权重显存减半（4B: 14.5→7.3 GB；MoE 层 2416→1208 MB），每量化层前向回到原生 Linear 水平（单模块 1.44×→0.99×；整层 MoE 1.35–1.39×→0.97–1.04×）。
3. **MoE 前向**（见 `03`）：把逐专家 Python 循环 + `.nonzero()` 同步换成"GPU 排序分桶 + 融合 gate/up 的 2 次 `bmm`"，单层 **1.77–22.8×**；fp32 相对误差 ~1e-6（算法等价）。默认走精确桶，另有带硬上限的 sync-free 可选。
4. **端到端**（只量化 layer0/48）：指标与旧实现**完全一致**（hellaswag 0.625 / piqa 0.875 / winogrande 0.750）；单次前向 1.02–1.06×（prefill）。要看端到端大收益需量化更多层。
5. **昇腾**（见 `06`）：`torch.bmm` **支持**（torch_npu 的 op_plugin 映射到 CANN BatchMatMul），`torch._grouped_mm` 不支持。昇腾专属风险点（strided B 格式、batch 并行度、uint8、`.item()`、bf16、版本矩阵）已列成待验证清单；`ascend_op_probe.py` 到卡上一跑即可定实现。
6. **无卡也能验证**（见 `05`）：正确性与设备无关，两套自检在纯 CPU 上 **23/23** 与 **18/18** 通过。

## 环境与硬件

| 项 | 值 |
|---|---|
| conda 环境 | `skw_quant_env` |
| torch / transformers | 2.8.0+cu128 / 5.9.0 |
| 本机加速器 | NVIDIA A800-SXM4-80GB（SM80, cc 8.0） |
| 昇腾 | **本机无**（无 `npu-smi`、无 `torch_npu`）→ 昇腾结论均为待验证 |
| 目标模型 | Qwen3-30B-A3B-Instruct-2507（48 层 / 128 专家 / top-8）、Qwen3-4B（dense 回归对照） |

## 改动清单

| 文件 | 类型 | 内容 | 验证 | 状态 |
|---|---|---|---|---|
| `LiftQuant/quantize/tmplinear.py` | 改 | `materialize(dtype=None)`：缓存权重按目标 dtype 一次性转换（fp32→bf16） | 旧行为逐位相同 / 与 `-layer1,2.dequant.pth` bit-exact / 4B 指标不变 | ✅ 已验证 |
| `LiftQuant/models/moe_fast_eval.py` | 新 | 向量化 MoE（`FastMoEExperts`/`FastMoEBlock`）+ 结构转换与释放 + 三道守卫 | CPU 自检 23/23；真实 layer0 A/B 1.77–22.8×；fp32 等价 ~1e-6 | ✅ 已验证（CUDA） |
| `LiftQuant/device_utils.py` | 新 | 后端抽象（懒加载 `torch_npu`、`resolve()` 返回字符串、可读报错） | stub 自检 18/18；CUDA 指标不变 | ✅ 逻辑已验证；真机待验 |
| `LiftQuant/e2e_utils.py` | 改 | `fast_moe` / `moe_sync_free_row_budget` 接入；`device_count()`/`empty_cache()` | 30B/4B lm-eval A/B | ✅ 已验证 |
| `allq/eval/eval_quant_lmeval.py` | 改 | `--fast-moe/--no-fast-moe`、`--moe-sync-free-row-budget`、`--device auto`，写入结果 JSON | 同 A/B | ✅ 已验证 |
| `allq/scripts/eval_by_lmeval.sh` | 改 | `FAST_MOE`、`MOE_SYNC_FREE_BUDGET`、`EVAL_DEVICE`、`ASCEND_RT_VISIBLE_DEVICES` | 冒烟 + shell 语法 | ✅ 已验证 |
| `allq/tools/moe_fast_selfcheck.py` | 新 | **无卡**正确性自检（23 项） | 23/23 | ✅ 已验证 |
| `allq/tools/device_utils_selfcheck.py` | 新 | **无卡**后端抽象自检（18 项，stub NPU） | 18/18 | ✅ 已验证 |
| `allq/tools/ascend_op_probe.py` | 新 | 昇腾算子/性能探测（CUDA 与 NPU 同一份脚本） | CUDA 侧已跑通 | ⏳ 真机待跑 |

未改动：`auto-round/`（仓库规则要求先征得同意）、`chat/`（仍绑 bitblas + `packed_weight`）、量化训练主流程。

## 关键命令速查

```bash
# 无卡自检（任意机器）
python allq/tools/moe_fast_selfcheck.py            # 23/23
python allq/tools/device_utils_selfcheck.py        # 18/18

# 算子探测（本机 CUDA / 昇腾真机）
python allq/tools/ascend_op_probe.py --device cuda --json /tmp/probe_cuda.json
python allq/tools/ascend_op_probe.py --device npu  --json /tmp/probe_910.json

# 评测（默认 fast MoE 开）
./run_eval.sh
FAST_MOE=0 TASKS=hellaswag LIMIT=8 ./allq/scripts/eval_by_lmeval.sh   # A/B 关掉
EVAL_DEVICE=npu:0 ... ./allq/scripts/eval_by_lmeval.sh                # 昇腾（待真机）
```

## 术语

| 术语 | 含义 |
|---|---|
| `fused layout` | 把 E 个专家权重堆成 `gate_up (E,2I,H)` / `down (E,H,I)`，与 HF `Qwen3MoeExperts` 一致 |
| `bucket` / `tmax` | 每个专家 padding 到的行数；`(E,tmax,H)` 是 bmm 的输入 |
| 精确桶 | `tmax = max(counts)`，需 1 次 `.item()` 主机同步 |
| sync-free 桶 | 用数据无关上界（`bound`），零同步但要 padding 更多 |
| `assume_distinct_topk` | 声明 router 每 token 返回互斥专家（`torch.topk` 语义）⇒ `bound = num_tokens` |
| guard-rail | `MAX_PADDED_ROWS` 超限时切到逐专家回退循环（正确但慢） |

## 文档索引

| 文档 | 内容 |
|---|---|
| `01_root_cause_eval_speed.md` | 评测慢的根因（fp32 提升、每次前向重铸、逐专家循环与同步）+ 过期缓存事故 |
| `02_materialize_dtype_fix.md` | 修复一：`materialize(dtype=)` 的设计、逐位证据与收益 |
| `03_moe_vectorized_forward.md` | 向量化 MoE：设计、决策（桶/守卫/`assume_distinct_topk`）、实测、集成、端到端 A/B、踩坑 |
| `04_device_abstraction.md` | `device_utils`：API、两条坑、接线与验证 |
| `05_verification_playbook.md` | 三层验证模型、三个自检工具、A/B 协议、复现命令、证据索引 |
| `06_ascend_porting_plan.md` | bmm 结论与依据、昇腾差异清单、融合算子候选、Phase D/E、纪律 |
| `log.md` | 时序开发日志：决策、踩坑与修正、被否决方向、遗留问题 |
