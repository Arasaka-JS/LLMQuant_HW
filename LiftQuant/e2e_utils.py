from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
import torch
from tqdm import tqdm
import gc
import re
import os
from quantize.tmplinear import TmpLinear, FWTLinear, make_fwtlinear_eval, build_moe_grouped_fwt_eval
from models.moe_fast_eval import convert_moe_blocks_to_fast, SYNC_FREE_ROW_BUDGET as MOE_SYNC_FREE_BUDGET
from device_utils import device_count, empty_cache


def resolve_torch_dtype(dtype):
    if dtype in (None, "auto"):
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_map[dtype]

def _is_qwen3_moe(config):
    """Return True when ``config`` describes a Qwen3-MoE architecture."""
    model_type = getattr(config, "model_type", "") or ""
    architectures = getattr(config, "architectures", None) or []
    return model_type == "qwen3_moe" or any(
        "moe" in (arch or "").lower() for arch in architectures
    )


def _get_moe_group_count(state_dict):
    """Infer the number of shared rotation/scale groups from a grouped checkpoint."""
    max_g = -1
    for key in state_dict:
        if "moe_shared." not in key:
            continue
        rest = key.split("moe_shared.", 1)[1]
        try:
            g = int(rest.split(".", 1)[0])
        except ValueError:
            continue
        max_g = max(max_g, g)
    return max_g + 1 if max_g >= 0 else 0


def _peek_moe_group_count_per_layer(quant_model_path):
    layer0_path = f"{quant_model_path}-layer0.pth"
    if not os.path.exists(layer0_path):
        return 0
    state_dict = torch.load(layer0_path, map_location="cpu")
    return _get_moe_group_count(state_dict)


def _replace_moe_layer_for_eval(layer, wbits, expc, num_groups):
    """Rebuild one MoE decoder layer into the grouped FWTLinear eval structure."""
    # Attention projections stay dense (per-projection rotation/scale).
    for name, module in get_named_linears(layer.self_attn, torch.nn.Linear).items():
        with torch.no_grad():
            fwt = make_fwtlinear_eval(module, wbits, expc, device="cpu", dtype=torch.float16)
            set_op_by_name(layer.self_attn, name, fwt)
    # MoE experts use group-shared rotation/scale.
    build_moe_grouped_fwt_eval(layer.mlp, wbits, expc, num_groups, device="cpu", dtype=torch.float16)


def get_named_linears(module, type):
    # return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_packed_quantized_layer_indices(state_dict):
    pattern = re.compile(r"^model\.layers\.(\d+)\..*\.packed_weight$")
    return sorted(
        {
            int(match.group(1))
            for key in state_dict
            if (match := pattern.match(key)) is not None
        }
    )

def check_meta_tensors(model, context: str = "当前状态"):
    """
    遍历一个 PyTorch 模型的所有参数 (parameters) 和缓冲区 (buffers)，
    检查是否有任何张量位于 'meta' 设备上，并打印详细信息。

    参数:
    - model (nn.Module):需要检查的 PyTorch 模型。
    - context (str): 一个描述性字符串，用于说明当前是在哪个代码阶段进行检查。
    """
    print(f"\n--- 检查模型中的 Meta 张量 ({context}) ---")
    
    found_meta_tensor = False

    # 1. 检查模型参数 (Parameters)
    for name, param in model.named_parameters():
        print(name, param.shape, param.dtype)
        if param.device.type == 'meta':
            print(f"[参数 - Meta] 位于 'meta' 设备: {name}")
            found_meta_tensor = True

    # 2. 检查模型缓冲区 (Buffers)
    # 缓冲区通常用于存储非训练参数，比如 BatchNorm 的 running_mean
    for name, buf in model.named_buffers():
        print(name, param.shape, param.dtype)
        if buf.device.type == 'meta':
            print(f"[缓冲区 - Meta] 位于 'meta' 设备: {name}")
            found_meta_tensor = True

    if not found_meta_tensor:
        print(">>> 结论: 所有参数和缓冲区都在具体的物理设备上 (非 'meta' 设备)。模型已正确具象化。")
    else:
        print(">>> 结论: 发现 'meta' 张量！模型尚未完全加载权重或未正确移动到设备上。")
        
    print(f"--- 检查完成 ({context}) ---\n")

    return found_meta_tensor

def load_quantized_model(fp_model_path, quant_model_path, wbits, expc, w_ternary, load_per_layer, auto_mix_precision = False, eval_dtype = "float32", fast_moe = True, moe_sync_free_row_budget = MOE_SYNC_FREE_BUDGET):
    print(f"Loading quantized model from {fp_model_path}")

    state_dict = None
    if not load_per_layer:
        state_dict = torch.load(quant_model_path, map_location='cpu')
        quantized_layer_indices = get_packed_quantized_layer_indices(state_dict)
        if not quantized_layer_indices:
            raise ValueError(
                "LiftQuant checkpoint contains no packed quantized layers. "
                "Run Stage2 with --finetuning_weights or convert the checkpoint to packed FWTLinear format."
            )
        for key in list(state_dict.keys()):
            if key.endswith('.packed_weight'):
                state_dict[key] = state_dict[key].flatten()
        print(f"Detected packed quantized layers: {quantized_layer_indices}")

    # import pdb;pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(fp_model_path, use_fast=False)
    config = AutoConfig.from_pretrained(fp_model_path)
    is_moe = _is_qwen3_moe(config)
    if is_moe:
        from models.qwen3_moe_per_expert import patch_qwen3_moe_per_expert
        patch_qwen3_moe_per_expert()
    if load_per_layer:
        # 逐层/部分量化：先加载完整 FP 权重，再只覆盖量化层（未量化层保持 FP）
        model = AutoModelForCausalLM.from_pretrained(
            fp_model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, trust_remote_code=True)
    else:
        # 全量化（单 .pth）：保持 meta 空壳 + 整体加载（与现状一致，不额外加载 FP）
        with init_empty_weights(): # 生成空的占位模型
            model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)

    num_groups = 1
    if is_moe:
        if state_dict is not None:
            num_groups = _get_moe_group_count(state_dict)
        else:
            num_groups = _peek_moe_group_count_per_layer(quant_model_path)
        num_groups = max(num_groups, 1)
        print(f"Detected MoE shared group count: {num_groups}")
    #if load_per_layer:
    #    # 加载模型的非layer权重
    #    non_layer_state_dict = torch.load(quant_model_path+'-non_layer.pth', map_location="cpu")
    #    model.load_state_dict(non_layer_state_dict, strict=False)
    #    print(non_layer_state_dict)
    #    print(model.model.norm.weight)
    #    model.model.norm = model.model.norm.to('cpu')
    #    print(model.model.norm.weight)
    layers = model.model.layers
    expc_choice = None
    if auto_mix_precision:
        num_layers = len(layers)
        if 'llama-3' in fp_model_path.lower():
            llama3_choice = [2, 2, 1, 2, 3, 3, 1, 3, 3, 2, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3,
                                    1, 2, 2, 3, 1, 2, 2, 3, 1, 1, 1, 3, 1, 1, 2, 3, 1, 1, 1, 2, 1, 1,
                                    1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2,
                                    1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                                    1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2,
                                    1, 2, 2, 2, 1, 2, 3, 2, 1, 2, 3, 3, 2, 3, 3, 3, 3, 3]
            if len(llama3_choice) == 4 * num_layers:
                expc_choice = llama3_choice
            else:
                print(
                    f"[e2e_utils] llama-3 auto_mix_precision schedule has {len(llama3_choice)} "
                    f"entries but this model has {num_layers} layers (expects {4 * num_layers}); "
                    "falling back to uniform expc."
                )
        else:
            print(
                "[e2e_utils] auto_mix_precision has no per-layer schedule for this model family; "
                "falling back to uniform expc."
            )
    if load_per_layer:
        layers_to_replace = [
            i for i in range(len(layers))
            if os.path.exists(f'{quant_model_path}-layer{i}.pth')
        ]
        if not layers_to_replace:
            raise ValueError(f"No per-layer LiftQuant checkpoints found for prefix {quant_model_path}")
    else:
        layers_to_replace = quantized_layer_indices
    invalid_layers = [i for i in layers_to_replace if i >= len(layers)]
    if invalid_layers:
        raise ValueError(
            f"Quantized layer indices {invalid_layers} exceed model layer count {len(layers)}"
        )
    # Split quantized layers into those that already have a dequantized FP cache
    # (load directly, skipping the expensive FWTLinear rebuild + materialize) and
    # those that still need the full pipeline.
    if load_per_layer:
        cached_layers = [
            i for i in layers_to_replace
            if os.path.exists(f'{quant_model_path}-layer{i}.dequant.pth')
        ]
        missing_layers = [i for i in layers_to_replace if i not in cached_layers]
    else:
        cached_layers = []
        missing_layers = layers_to_replace
    for i in tqdm(missing_layers):
        layer = layers[i]
        if is_moe:
            _replace_moe_layer_for_eval(layer, wbits, expc, num_groups)
            continue
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():

            with torch.no_grad():
                if expc_choice!= None:
                    if "q_proj" in name or "q_proj" in name  or "q_proj" in name:
                        expc = expc_choice[i*4]
                    if "o_proj" in name:
                        expc = expc_choice[i*4+1]
                    if "gate_proj" in name or "up_proj" in name :
                        expc = expc_choice[i*4+2]
                    if "down_proj" in name:
                        expc = expc_choice[i*4+3]
                    if expc ==0:
                        expc = 'nl'
                    if expc ==1:
                        expc = 'nm'
                    if expc ==2:
                        expc = 'np'
                    if expc ==3:
                        expc = 'nh'
                fake_linear = torch.nn.Linear(module.in_features,module.out_features,not module.bias is None, device = 'cuda', dtype = torch.float16) 
                #convert to tmplinear
                tmplinear = TmpLinear(fake_linear, wbits, expc = expc, training_trans = True)
                tmplinear.find_params()
                tmplinear.quantizer.alpha = torch.nn.Parameter(0.*torch.ones(tmplinear.quantizer.scale.shape, device = tmplinear.orilinear.weight.device , dtype = tmplinear.orilinear.weight.dtype ))
                
                # conert to adalinear
                '''adalinear= AdaLinear()
                adalinear.convert_form_tmplinear(tmplinear, maxq=2, expc=expc, training_trans =True)
                del tmplinear, fake_linear
                # convert to fuse buffer
                adalinear.remove_adaquant()
                adalinear = adalinear.to('cpu')
                set_op_by_name(layer, name, adalinear)'''

                # convert to FWTLinear
                fwtlinear =  FWTLinear()
                #fwtlinear.convert_form_tmplinear(tmplinear, expc = expc, training_trans = True, groupsize=-1 )
                fwtlinear.convert_form_tmplinear(tmplinear, bits=wbits, expc=expc, training_trans = True, groupsize = -1)
                fwtlinear.bit_channel_convert(True)
                fwtlinear.pack_to_int8()
                fwtlinear = fwtlinear.to('cpu')
                set_op_by_name(layer, name, fwtlinear)
                #if load_per_layer:
                #    #model.model.layers[i]
                #    layer.load_state_dict(torch.load(quant_model_path+'-layer'+str(i)+'.pth', map_location="cpu"), strict=False)

        #print(model.model.layers[0].mlp.up_proj.Trans.linear_u_left.dtype)
    #print(model.model.layers[0].mlp.up_proj.Trans.linear_u_left.dtype)
    empty_cache()
    gc.collect()
    model.tie_weights()
    
    # For layers that already have a dequantized FP cache, load it straight into
    # the FP model's nn.Linear weights (no FWTLinear structure was built for them).
    for i in cached_layers:
        cache = torch.load(f'{quant_model_path}-layer{i}.dequant.pth', map_location='cpu')
        model.load_state_dict(cache, assign=True, strict=False)
    if cached_layers:
        print(f"Loaded dequantized FP cache for {len(cached_layers)} layers.")
    
    print("Loading pre-computed quantized weights...")
    
    if load_per_layer:
        # embed_tokens/norm/lm_head 已由 from_pretrained 加载（均为未量化 FP），无需 non_layer.pth；
        # 只把「存在量化文件」的层覆盖成量化权重，其余层保持 FP。
        for i in missing_layers:
            layer_path = f'{quant_model_path}-layer{i}.pth'
            state_dict = torch.load(layer_path, map_location="cpu")
            for key in list(state_dict.keys()):
                if key.endswith('.packed_weight'):
                    state_dict[key] = state_dict[key].flatten()

            model.model.layers[i].load_state_dict(state_dict, assign=True, strict=False)
    else: #分支2
        '''for param_name, tensor in state_dict.items():
            print(f"参数名称 (Key): {param_name}")
            print(f"  - 形状 (Shape): {tensor.shape}")
            print(f"  - 数据类型 (Dtype): {tensor.dtype}")
            print(f"  - 所在设备 (Device): {tensor.device}")
            print("-" * 30)'''
        model.load_state_dict(state_dict, assign=True, strict=False)

    #check_meta_tensors(model)

    target_dtype = resolve_torch_dtype(eval_dtype)
    # Evaluation should follow the same dtype choice as eval_by_lmeval.sh: the
    # caller passes float16/bfloat16/float32, while "auto" keeps the checkpoint
    # dtypes instead of forcing the LiftQuant model to float32.
    if target_dtype is not None:
        model = model.to(target_dtype)
    # Pre-dequantize every FWTLinear once on CPU (before dispatch).  This is
    # bit-identical to dequantizing on each forward (get_weight is a pure
    # function) but avoids re-running unpack/scale/rotation per token.  Doing it
    # before infer_auto_device_map/dispatch means (a) the transient dequant
    # peak lives in host RAM (1TiB) instead of a single 80GB GPU, and (b) the
    # device map is computed on the fully-materialized ~56GB model so it shards
    # correctly instead of seeing only the ~31GB packed weights.
    materialized = 0
    if load_per_layer:
        # Materialize per missing layer and immediately persist its dequantized
        # FP weights, so even if dispatch/eval fails later we can reuse the cache.
        for i in missing_layers:
            with torch.no_grad():
                for module in model.model.layers[i].modules():
                    if isinstance(module, FWTLinear):
                        module.materialize()
                        materialized += 1
            dequant = {}
            for rel_name, module in model.model.layers[i].named_modules():
                if isinstance(module, FWTLinear):
                    dequant[f"model.layers.{i}.{rel_name}.weight"] = module._weight_fp.detach().cpu()
                    if module.bias is not None:
                        dequant[f"model.layers.{i}.{rel_name}.bias"] = module.bias.detach().cpu()
            torch.save(dequant, f'{quant_model_path}-layer{i}.dequant.pth')
            if fast_moe:
                # Swap the per-expert MoE blocks of this layer for the vectorized
                # fused implementation.  Done after the dequant cache write (which
                # reads `_weight_fp`) and before dispatch, so the fused buffers are
                # moved to the right devices together with the rest of the layer.
                converted = convert_moe_blocks_to_fast(
                    model.model.layers[i], dtype=target_dtype,
                    sync_free_row_budget=moe_sync_free_row_budget,
                )
                if converted:
                    print(f"[fast_moe] layer {i}: fused {len(converted)} MoE block(s)")
    else:
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, FWTLinear):
                    module.materialize()
                    materialized += 1
        if fast_moe:
            converted = convert_moe_blocks_to_fast(
                model, dtype=target_dtype, sync_free_row_budget=moe_sync_free_row_budget,
            )
            if converted:
                print(f"[fast_moe] fused {len(converted)} MoE block(s): "
                      f"vectorized sort+bmm expert forward")
    if materialized:
        print(f"Materialized {materialized} quantized linear modules to FP cache.")

    # Manually shard the ~57GB materialized model across GPUs at whole-layer
    # granularity.  infer_auto_device_map's automatic balancing misbehaves on
    # the per-expert MoE structure (it offloads modules to CPU even when they
    # would fit), so we build a deterministic round-robin layer map instead.
    num_gpus = device_count() or 1
    num_layers = len(model.model.layers)
    device_map = {}
    for i in range(num_layers):
        device_map[f"model.layers.{i}"] = i % num_gpus
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = (num_layers - 1) % num_gpus
    if hasattr(model.model, "rotary_emb"):
        device_map["model.rotary_emb"] = 0
    device_map["lm_head"] = (num_layers - 1) % num_gpus
    model = dispatch_model(model, device_map=device_map)
    print("Loaded quantized weights successfully.")
    empty_cache()
    gc.collect()
    #load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    #print("Loading pre-computed quantized weights Successfully")
    #print(model.model.layers[0].mlp.up_proj.Trans.linear_u_left.dtype)
    return model, tokenizer
