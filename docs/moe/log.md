# Qwen3-30B-A3B 适配 LiftQuant —— 问题清单（log）

> 目标模型：`checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507`
> 目标框架：`LiftQuant`（权重量化主线）

## 结论

主量化路径与配套加载/评估路径已完成 Qwen3-MoE 适配（见第一、二轮实施记录）；剩余项为可选路径 / 稳健性 / 环境相关，见「剩余问题清单」。核心方案：用 transformers 5.9.0 仍存在的 `Qwen3MoeMLP` + `Qwen3MoeTopKRouter`，把融合专家 block 猴子补丁回「逐专家 `nn.ModuleList[Qwen3MoeMLP]`」布局，使本地逐专家 checkpoint 原样加载。

## 已确认的模型信息

- `config.json`：`architectures: ["Qwen3MoeForCausalLM"]`、`model_type: "qwen3_moe"`
- `num_experts=128`、`num_experts_per_tok=8`、`moe_intermediate_size=768`
- `hidden_size=2048`、`num_hidden_layers=48`、`num_attention_heads=32`、`num_key_value_heads=4`、`head_dim=128`
- 权重索引：`model.layers.{i}.mlp.experts.{0..127}.gate_proj/up_proj/down_proj.weight` + `model.layers.{i}.mlp.gate.weight`（路由器），**无顶层 `mlp.up_proj/down_proj`**

## 问题清单（注释掉的不需要处理）

### 1. 资源门槛高
- 位置：`LiftQuant/models/LMClass.py:38`（`device_map='cpu'` 一次性加载）
- 现象：30B fp16 约 60GB+ 内存（第三轮实测 CPU RSS 峰值 ~63GB，本机 1TiB 内存无压力）；逐层搬 GPU 时单层含 128×768 的 MoE 也很大，需 `--save_per_layer`、减小 `--nsamples` 等控制峰值。

### 2. 配套路径同样不支持 MoE（已解决，见第二轮）
- `LiftQuant/e2e_utils.py:144`（`get_named_linears(layer, torch.nn.Linear)`）假设稠密；`auto_mix_precision` 的 `expc_choice` 只给 llama-3。
- `LiftQuant/main.py` 的 `cache_name` 无 Qwen3-MoE 条目（会落到 `qwen3`）。

### 3. norm fusion 破坏 MoE 路由（已解决，见第二轮）
- 位置：`LiftQuant/quantize/liftq.py:235`
- 现象：守卫是 `if 'qwen3.' not in args.net.lower():`，而 `args.net = "Qwen3-30B-A3B-Instruct-2507"` 小写为 `qwen3-...`（不含 `qwen3.`），因此 norm fusion **会执行**。
- 它把 `post_attention_layernorm` 融合进所有 `up_proj`/`gate_proj`（含 128 个专家）并把 norm 置 1；但路由器 `mlp.gate` 是 `Qwen3MoeTopKRouter`（`weight` 是 `nn.Parameter`，不是 `nn.Linear`）**没被融合**。路由器和专家看到不同输入 → top-k 选专家错误 → 校准 `fp_outs`/量化权重/评估结果全错。
- 修复方向：`if not is_moe and 'qwen3.' not in args.net.lower():`。

conda 环境：skw_quant_env

## 实施记录（第一轮）

### 已解决（已移出清单）：触发逻辑、稠密分支崩溃、替换逻辑白名单、MoE 逐专家布局跑通、transformers 5.9.0 适配（不降级）

核心方案：**用 transformers 5.9.0 仍存在的 `Qwen3MoeMLP` + `Qwen3MoeTopKRouter`，把融合专家 block 猴子补丁回“逐专家 `nn.ModuleList[Qwen3MoeMLP]`”布局**。这样本地逐专家 checkpoint 原样加载，LiftQuant 现有 `experts[0]/experts[1]` MoE 分支不用改。

改动文件：

1. 新增 `LiftQuant/models/qwen3_moe_per_expert.py`
   - `PerExpertQwen3MoeSparseMoeBlock`：`experts = nn.ModuleList([Qwen3MoeMLP(...)])` + `gate = Qwen3MoeTopKRouter(...)`，forward 用 Mixtral 式逐专家循环。
   - `patch_qwen3_moe_per_expert()`：把 `modeling_qwen3_moe.Qwen3MoeSparseMoeBlock` 替换成上面这个类。

2. `LiftQuant/models/LMClass.py`
   - 导入并调用 `patch_qwen3_moe_per_expert()`（在 `from_pretrained` 之前）。

3. `LiftQuant/quantize/liftq.py`
   - 顶部新增 `is_moe`（基于 `model_type` / `architectures` 判断，不再只看目录名）。
   - `:276`、`:356` 的 `'moe' in args.net.lower()` 改为 `is_moe`。

### 验证结果（skw_quant_env, transformers 5.9.0）

- `py_compile` 三个文件通过。
- `patch` 后 `from_config(meta)`：layer0.mlp 的 key 为 `experts.{0..127}.gate_proj/up_proj/down_proj.weight`（768×2048 / 2048×768）+ `gate.weight`（128×2048），共 385 个 key。
- key 级对比：模型期望 18867 个 key 与 checkpoint 权重索引 18867 个 key **完全一致（missing=0, unexpected=0）**。
- 数值正确性：`PerExpertQwen3MoeSparseMoeBlock.forward` 与融合 `Qwen3MoeExperts` 参考实现输出 **max abs diff = 0.0**。
- `import models.LMClass / quantize.liftq` 通过（仅 bitblas 的 “CUDA extension not installed” 警告）。

## 实施记录（第二轮）

### 已解决：问题 2（e2e 加载路径 + cache_name）、问题 3（norm fusion）

1. `LiftQuant/e2e_utils.py`
   - 新增 `_is_qwen3_moe(config)`（`model_type == "qwen3_moe"` 或 `architectures` 含 `moe`）。
   - `from_config` 前对 Qwen3-MoE 调用 `patch_qwen3_moe_per_expert()`，使 meta 模型回到逐专家 `nn.Linear` 布局，`get_named_linears(nn.Linear)` 才能找到 q/k/v/o + 128×3 专家投影（路由器 `mlp.gate` 为 `nn.Parameter`，不会被误替换）。
   - 重写 `auto_mix_precision` 的 `expc_choice`：llama-3 仅在 `4×num_layers` 匹配时用现成表；其余（含 Qwen3-MoE 及 >32 层稠密模型）回退到统一 `expc` 并打印警告（方案：回退统一 expc + 警告）。

2. `LiftQuant/main.py`
   - `cache_name` 用 `lm.model.config`（`model_type`/`architectures`）判 MoE，命中即 `qwen3-moe`，置于通用 `qwen2`/`qwen3` 之前，避免目录名 `Qwen3-30B-A3B-Instruct-2507` 落到 `qwen3`。

3. `LiftQuant/quantize/liftq.py`
   - norm fusion 守卫改为 `if is_moe: skip / elif 'qwen3.' not in args.net.lower(): fuse`，MoE 不再折叠 RMSNorm，路由器 `mlp.gate` 保持原样，避免 top-k 路由被破坏。

4. `LiftQuant/quantize/stage_training.py`
   - `_build_worker_qlayer` 在 `from_config` 前对 Qwen3-MoE 调用 `patch_qwen3_moe_per_expert()`，使 DDP worker 重建的 layer 与主进程同为逐专家 `nn.Linear` 布局，`replace_linear_with_TmpLinear` 才能替换到专家投影。

### 验证结果

- `py_compile e2e_utils.py main.py` 通过。
- 冒烟：`_is_qwen3_moe` → True；patch 后 `from_config` + `get_named_linears` 返回 `self_attn.q_proj` + `mlp.experts.{0..3}.gate/up/down_proj`，不含 `mlp.gate`。
- `cache_name`：`Qwen3-30B-A3B-Instruct-2507` → `qwen3-moe`；稠密 `Qwen3.5-4B` → `qwen3.5`（无误判）。
- norm fusion 守卫：`Qwen3-30B-A3B`（is_moe）→ SKIP；`Qwen3.5-4B` → SKIP；`Llama-2-7b` → RUN（非 MoE/qwen3 行为不变）。
- DDP worker：patch 后 `from_config` 得到 `PerExpertQwen3MoeSparseMoeBlock`，`experts.{0..3}.gate/up/down_proj` 为 `nn.Linear`（`py_compile stage_training.py` 通过）。

## 实施记录（第三轮）：GPU2 实跑验证（layer 0，无 DDP）

### 运行配置
- 脚本：`run_moe_quant.sh`（`CUDA_VISIBLE_DEVICES=2`、单进程 `python main.py`、无 `--quant_training_ddp`）。
- `--quant_layers 0`、`--save_per_layer`、`--nsamples1/2 32`、`--expc 24to8`、`--wbits 2`、`--epochs1/2 1`、`--batch_size 1`、`--calib_dataset redpajama`。
- 未加 `--finetuning_weights`（只跑 Stage1 / block correction）。

### 结果：主量化管线对 MoE 第 0 层跑通
- 模型加载：逐专家布局 18867 keys 正常（约 15 分钟，RSS 峰值 ~63GB，57GB 磁盘读取）。
- 标定数据 redpajama 离线缓存命中。
- norm fusion 正确跳过（`=== Skip norm fusion for MoE (router gate must stay unfused) ===`）。
- `get_act_means` 通过（专家 0/1 被激活，无 KeyError）。
- `replace_linear_with_TmpLinear`：388 个 Linear（q/k/v/o + 128×3 专家）替换成功。
- a1init / Stage1 训练通过；保存 `layer0.pth`(1.2G) + `non_layer.pth`(1.2G)。
- 全程无 error / KeyError / OOM；GPU2 峰值 ~35GB。

### 新发现问题（已并入「剩余问题清单」）
1. Stage1-only 保存为 TmpLinear 格式（`orilinear.weight`/`quantizer.*`，`packed_weight=0`），离线评估 `load_quantized_model` 无法加载。
2. `main.py` 的 `print("Haven't set save_dir")` 误导。

## 剩余问题清单（第二轮排查，按严重程度）

- 【中】`LiftQuant/quantize/liftq.py:282/363`：MoE 分支 hook `experts.0.up_proj`/`experts.1.up_proj`，若标定数据里专家 0/1 未被路由选中，`act_disturb['experts.0.up_proj']` 缺失 → KeyError（仅对 NaN 做 fallback）。
- 【中】`LiftQuant/quantize/liftq.py:320`：`auto_mix_precision` 量化侧 `expc_list` 未定义 → `NameError`（评估侧已回退，量化侧未修；别开该 flag）。
- 【中】`LiftQuant/main.py:544`：`--save_per_layer` 下仅保存 Stage1 的 TmpLinear 格式（`orilinear.weight`/`quantizer.*`，无 `packed_weight`），离线评估 `load_quantized_model` 无法加载；需 `--finetuning_weights`（Stage2）转 FWTLinear + `pack_to_int8`。
- 【中】`LiftQuant/e2efinetune.py:20`：import 缺失的 `datautils_block`（e2e 微调 import 失败）。
- 【低】`LiftQuant/main.py:239`：`--expc` 默认 `"n"` 无效（`TmpLinear` 需 `"XtoY"`，量化需显式 `--expc 20to8`）。
- 【低】`LiftQuant/main.py:557`：`print("Haven't set save_dir")` 误导（`--save_per_layer` 已保存成功仍打印）。
- 【低】问题 1（资源）：30B 一次性全量加载进 CPU 内存（实测 ~63GB，本机 1TiB 无压力；57GB 磁盘读取约 15 分钟），未从代码层面解决。



# 9.8 开发计划：MoE 专家分组旋转矩阵（已实施，见第四轮实施记录）

## 目标
让 MoE 专家使用「分组旋转矩阵」：多个专家组成一组，每组 3 个旋转矩阵（gate/up/down 各一个）供组内所有专家共用；`up/gate` 的缩放系数用该组「被路由选中最多」的专家的激活统计来初始化。除 quant 参数（`quantizer.scale/alpha`、FWTLinear 的 `weight/scale/maxq/root/packed_weight`）逐专家外，**旋转矩阵和缩放矩阵（`Trans`、`a1`、`a2`）均为组级共享**。训练形态仍是 `TmpLinear`；先用 1 组跑通，再调优组数。

## 为什么 Stage1→Stage2 要把 TmpLinear 转成 FWTLinear
- `TmpLinear`（Stage1）是训练代理：保留原始浮点 `orilinear.weight`，每次前向用 `quant_tmpweight()` 现场重构「旋转 + 缩放 + 标量量化」权重，**不真正存量化权重、无压缩收益**。训练对象是 `Trans.linear_*`、`a1/a2`、`quantizer.scale/alpha`。
- `FWTLinear`（Stage2）是部署形态：物化旋转+缩放后的 `weight`，切到**向量/格点量化**（`bit_channel_convert` 读 `./lattice/{expc}.pt`），把格点嵌入 M 复合进 `Trans.linear_right`，最终 `pack_to_int8()` 打出紧凑 `packed_weight`。`load_quantized_model` / `chat_quant_bitblas` 只认这个 packed 形态。
- 结论：Stage1-only 保存不可评测（对应剩余问题清单中「TmpLinear 格式无法加载」），必须 Stage2 转 FWTLinear + pack。

## 为什么 MoE 不能照搬 dense 的转换
dense 里每个投影是独立 `TmpLinear`，各自持有 `Trans/a1/a2`，`replace_TmpLinaer_with_FWTLinear` 逐个叶子独立替换（每个都 `to_buffer()`）。MoE 分组后 `Trans/a1/a2` 组级共享，直接套用会：
1. `to_buffer()` 被 128 个专家各调一次 → 第二次起 `linear_u_left` 已被删除 → `AttributeError`；`bit_channel_convert` 里 `linear_right = M @ linear_right`、`a2 = a2.repeat_interleave(root2)` 被重复修改，语义错乱。
2. 若每个专家各存一份共享参数，就失去「节省矩阵数量」的意义。

因此转换必须把**共享物化提升到组级（每组建一次）**，**逐专家部分（weight/scale/maxq/root/格点搜索/pack）留在每个 FWTLinear**。

## 方案设计
- 新增 `MoeSharedRotScale(nn.Module)`，**拥有**每组每投影的 `Trans`、`a1`、`a2`，挂在 MoE block 上（例如 `mlp.moe_shared.{g}.{gate_proj,up_proj,down_proj}`），state_dict 只存一份。
- 每个专家的 `TmpLinear`/`FWTLinear` **引用**共享对象（用 `object.__setattr__` 绕过 `nn.Module.__setattr__`，避免重复注册导致 `named_parameters()`/`state_dict` 重复）。
- `quantizer.scale/alpha` 与 FWTLinear 的 `weight/scale/maxq/root/packed_weight` 仍逐专家。
- 组划分用连续专家索引均分；`num_groups=1` → 128 个专家一组。

## 文件级改动清单
1. `LiftQuant/quantize/tmplinear.py`
   - 新增 `MoeSharedRotScale`：`__init__(ic, oc, expc, training_trans, groupsize)` 计算 `transdim1/transdim2/root/root2/expic`，创建 `Trans`、`a1`(2-D `transdim1×transdim2`)、`a2`(`transdim1 × transdim2//root`)。
   - `TmpLinear.__init__` 增加可选 `shared=None`；给定时用 `object.__setattr__` 把 `a1/a2/Trans` 指向共享件，不自建；`quantizer/orilinear` 仍自建。
   - `find_params`/`quant_tmpweight` 的 `a1.data.reshape(...)` 加形状守卫（共享 a1 已是 2-D）。
   - 新增 `replace_linear_with_TmpLinear_moe(model, args, num_groups)`：识别 MoE block（`hasattr(m,'experts')` 且 `experts[0]` 有 `gate_proj`），按组建 `moe_shared`，专家投影替换为引用组内共享件的 `TmpLinear`；注意力 q/k/v/o 走原逻辑。
   - 新增 `replace_TmpLinaer_with_FWTLinear_moe(...)`（或给现有转换加 `shared` 分支）：每组每 proj_type 先对共享 `Trans` 调 `to_buffer()` 一次，并把 `M@linear_right`、`a2.repeat_interleave(root2)` 的共享物化只做一次；每个专家 `FWTLinear.convert_form_tmplinear(..., shared=组容器)`（`self.Trans` 已是 eval mode，不再 `to_buffer()`；`a1/a2` 引用共享件）。
2. `LiftQuant/quantize/liftq.py`
   - `is_moe` 处调用 `replace_linear_with_TmpLinear_moe(qlayer, args, args.moe_num_groups)`。
   - 新增 `get_moe_act_means`：hook `qlayer.mlp` 输入 hidden_states + `qlayer.mlp.gate` 输出 `selected_experts`，离线算每个专家 up_proj std 统计 + 路由计数（比 hook 128 个 Linear 省内存、规避「专家 0/1 未被路由 → KeyError」）。
   - `a1init` MoE 分支：每组取路由计数最多的专家，用其 up_proj 的 `std(dim=0)/std()`（clamp 1~16、pad 到 expic）赋给该组 gate/up 的共享 a1；down_proj 共享 a1 置 1。
3. `LiftQuant/quantize/stage_training.py`
   - `_prepare_stage1`：共享 `a1/a2` 被多个 TmpLinear 引用会重复 append → 按 `id(param)` 去重；`Trans.linear_*` 经 `named_parameters()` 天然去重。
   - `_prepare_stage2`：新增 MoE-grouped 分支——`weight/scale` 从 FWTLinear 逐专家收集，`a1/a2/linear_left/linear_right` 从 `MoeSharedRotScale` 组级收集并去重。
   - `_replace_tmp_with_fwt_structure` / `_convert_stage2_group`：MoE 时走 `replace_TmpLinaer_with_FWTLinear_moe`。
   - `_build_worker_qlayer`（DDP 重建）改用 `replace_linear_with_TmpLinear_moe`，保证 worker state_dict 结构与主进程一致；`_prepare_stage1_state_structure` 相应适配。
4. `LiftQuant/main.py`
   - 新增 `--moe_num_groups`（int，默认 1）；顺带修 `--expc` 默认 `"n"` 无效问题。
5. （后续必要 follow-up）`LiftQuant/e2e_utils.py`、`LiftQuant/chat/chat_quant_bitblas.py`
   - `load_quantized_model` 目前按 `nn.Linear` 逐个建 FWTLinear，state_dict 结构是 `experts.{i}.{proj}.Trans.*`；分组后保存 key 为 `mlp.moe_shared.{g}.{proj}.Trans.*`，会 unexpected/missing。需按组重建 `MoeSharedRotScale` 并把共享件引用进每个专家 FWTLinear（`strict=False` 才能正确加载）。不影响「先把量化流程跑通」，作为第二阶段。

## 关键难点
1. 共享物化只做一次：`to_buffer()`、`linear_right = M@linear_right`、`a2 = a2.repeat_interleave(root2)` 按 (group, proj_type) 各执行一次；逐专家只做 `l2/格点搜索/weight/scale/maxq/root/pack`。
2. 共享参数去重：遍历模块收集参数处（stage1 a1/a2、stage2 a1/a2/linear_*）须 `id()` 去重，否则 AdamW 报 duplicate params。
3. a1 初始化的「最多被选专家」需要路由计数，不能复用现有 `experts.0/1` hook。
4. 共享件用 `object.__setattr__` 挂到 TmpLinear/FWTLinear，保证 state_dict 只存一份、`named_parameters()` 不重复。
5. DDP worker 必须用同样的分组替换，否则广播 state_dict 对不上。

## 验证方式（先 1 组跑通）
- `py_compile` 改动文件。
- 冒烟：patch 后 layer0 调 `replace_linear_with_TmpLinear_moe(qlayer, args, num_groups=1)`，断言 `mlp.moe_shared.0.gate/up/down` 各有一套 `Trans/a1/a2`；128×3 专家 TmpLinear 的 `a1/a2/Trans is 组共享对象`；state_dict 中共享 key 只出现一次。
- 构造 `_prepare_stage1`/`_prepare_stage2` optimizer 不报 duplicate。
- 小规模实跑（`run_moe_quant.sh` + `--quant_layers 0 --moe_num_groups 1 --finetuning_weights`）确认 Stage1→Stage2 转换与 `pack_to_int8` 无报错。
- 数值：1 组（更强共享约束）loss 略高于逐专家独立 baseline 属预期。

## 假设与待确认
- 组划分用连续专家索引均分，`num_groups=1` → 128 专家一组。
- 「三个旋转矩阵」= 每组 gate/up/down 各一个（gate 与 up 不共享）；a1/a2 同理每组每投影各一套、组内专家共享。
- 注意力 q/k/v/o 不分组，保持 dense 行为。
- down_proj 的 a1 初始化为 1；仅 gate/up 用「最多被选专家」统计初始化。


## 实施记录（第四轮）：分组旋转矩阵实现 + GPU 实跑 + 评估适配

### 1. 训练侧：分组旋转矩阵（按 9.8 开发计划落地）
改动文件：`quantize/tmplinear.py`、`quantize/liftq.py`、`quantize/stage_training.py`、`quantize/utils.py`、`main.py`

- `tmplinear.py`
  - 新增 `MoeSharedRotScale`（组级共享容器，持有 `Trans/a1/a2`）。
  - `TmpLinear.__init__` 增 `shared=None`；`find_params` 的 `a1` reshape 加形状守卫。
  - `FWTLinear.convert_form_tmplinear` 增 `shared/to_buffer` 分支；新增 `bit_channel_convert_shared` + `prepare_moe_shared_for_export`（组级物化只做一次）。
  - 新增 `_is_moe_block`、`_replace_moe_block_grouped`、`replace_linear_with_TmpLinear_moe`、`_convert_moe_block_grouped`、`replace_TmpLinaer_with_FWTLinear_moe`。
  - 新增 `make_fwtlinear_eval`、`build_moe_grouped_fwt_eval`（评估侧结构重建）。
  - **关键修复**：FWTLinear 共享引用改用 `self.shared`（holder 模块）+ `_a1/_a2/_trans` 访问器；因 `load_state_dict(assign=True)` 会替换 Parameter 对象，原先 `self.a1 = shared.a1` 的引用会失效（实测 `fwt.a1 is holder.a1 → False`、值 stale），改动态访问 `self.shared.a1` 解决。
- `liftq.py`
  - 新增 `get_moe_act_means`（hook `mlp` 输入 + `mlp.gate` 输出，取每专家路由计数 + up_proj std 统计，规避原 experts.0/1 未路由的 KeyError）。
  - `replace_linear_with_TmpLinear_moe` 按 `--moe_num_groups` 调用；a1init 按组取「最多被选专家」统计初始化 gate/up 共享 a1，down 共享 a1 置 1。
- `stage_training.py`
  - `_prepare_stage1`：共享 `a1/a2` 按 `id()` 去重。
  - `_prepare_stage2`：结构化判 MoE（`MoeSharedRotScale` 或 `mlp.experts` 为 ModuleList），共享 `a1/a2/linear_*` 从 holder 收集。
  - `_convert_stage2_group` 分组分发；`_build_worker_qlayer` 用 `replace_linear_with_TmpLinear_moe`。
- `utils.py`：新增 `get_moe_act_means`。
- `main.py`：新增 `--moe_num_groups`（默认 1）；`--expc` 默认 `"n"` → `"24to8"`（修复剩余问题清单低优先项）。

### 2. GPU2 实跑（layer0，`--moe_num_groups 1 --finetuning_weights`）
- 全程 0 报错；Stage1 训练 + Stage2 分组转换 + 训练 + pack 全部跑通。
- 产物：`qmodels/Qwen3-30B-A3B-Instruct-2507/Qwen3-30B-A3B-Instruct-2507+24to8-layer0.pth`(235MB) + `...-non_layer.pth`(1.24GB)。
- checkpoint 结构验证：
  - `moe_shared` 键 12 个（1 组 × 3 投影 × {a1, a2, Trans.linear_left, Trans.linear_right}）。
  - `packed_weight` 388（4 注意力 + 384 专家）、专家 `scale` 384、逐专家 `a1/a2` 0。
  - 旋转+缩放从 `128×3×4=1536` 参数降到 12，节省 128×。
- 注意：Stage2 格点搜索（`find_nearest_fast`, fast=False）对 128 专家较慢（每专家投影 ~15-20s，共约 1.5-2h）。

### 3. 评估路径适配（`e2e_utils.py`）
- 新增 `_get_moe_group_count`（从 `moe_shared.{g}.` 键推断组数）、`_peek_moe_group_count_per_layer`、`_replace_moe_layer_for_eval`（注意力 dense + 专家 grouped）。
- `load_quantized_model`：检测 `is_moe` + 从 checkpoint 推断 `num_groups`，MoE 层走分组重建。
- 验证：真实 `layer0.pth` 结构重建后 `load_state_dict(assign=True, strict=False)` → **missing=0 / unexpected=0**；`fwt._a1()/_a2()/_trans() is holder.*` → True；`get_weight()` 形状 (768,2048) 有限。

### 遗留（新增）
- 【中】partial 量化评估：只量化 layer0 时，`load_quantized_model(load_per_layer=True)` 会遍历全部 48 层加载 `layer{i}.pth`，层 1~47 无文件会报错。需补「非量化层加载 FP 权重（fp_model_path 的 sharded safetensors，~60GB）+ 量化层加载 layer0.pth」逻辑。
- 【中】`chat/chat_quant_bitblas.py` 未适配分组 MoE（key 重映射仍按逐专家结构）；`run_eval.sh` 不依赖，可后置。
- 【低】`auto_mix_precision` 量化侧 `expc_list` 未定义 `NameError`（`liftq.py:320`）仍未修；MoE 量化别开该 flag。
