import torch
import torch.nn as nn
import torch.nn.functional as F
from models.int_llama_layer import QuantLlamaDecoderLayer
from quantize.tmplinear import *
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import  get_parameters, get_act_means, get_moe_act_means
from quantize.stage_training import null_traincast, run_stage1_training, run_stage2_training


from tqdm import tqdm

import functools
from scipy import linalg

from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import os


from scipy import linalg

# GPTQ
from gptq.gptq import *
from gptq.modelutils import *
from gptq.quant import *

from trans_utils import Hadamard_trans, ORTransMatrix, pca_cov, PCA_rotation

 
import torch.nn.functional as F

def get_n_set_parameters_byname(model, required_names):
    params = []
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return params

def get_n_set_parameters_byname_FWT(model, required_names):
    params = []
    for r_name in required_names:
        for n,m in model.named_modules():
            if isinstance(m, FWTLinear):
                for name, param in m.named_parameters():
                    if name.find(r_name) > -1:
                        params.append(param)
    for param in params:
        param.requires_grad = True
    return params

def print_trainable_parameters(model):      
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    print('trainable module')
    print('*'*80)
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(name, "is trainable")
            trainable_params += param.numel()
    print('*'*80)
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def liftq(
    lm,
    args,
    dataloader,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    #量化过程关闭cache
    model.config.use_cache = False
    is_moe = (
        'moe' in args.net.lower()
        or getattr(model.config, 'model_type', '') in ('qwen3_moe', 'qwen2_moe')
        or any('moe' in a.lower() for a in getattr(model.config, 'architectures', []))
    )
    is_llama = False
    if args.info:
        print(args)
        print(model)
        print(type(model))

    if "llama" in args.net.lower() or "qwen" in args.net.lower(): 
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        #下面这三行好像没啥用
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for qwen2.5, llama-2, Llama-3/3.1/3.2 now")
    
    
    
    if args.save_dir and args.save_per_layer :
        # 如果目录不存在则创建
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir,args.net, args.net+'+'+args.expc+'-non_layer.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        non_layer_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("model.layers.")}
        torch.save(non_layer_state_dict, save_path)
        print('save non-layer-statedict')
        #把模型名字中没有layer的部分都摘出来单独保存为一个pth
    
    explicit_quant_layers = args.quant_layers is not None
    args.quant_end = min(args.quant_end, len(layers))
    if explicit_quant_layers:
        quant_layer_indices = set(args.quant_layers)
        processing_end = max(quant_layer_indices) + 1
    else:
        quant_layer_indices = set(range(args.quant_start, args.quant_end))
        processing_end = args.quant_end
        for i in range(len(layers)):
            if i >= args.quant_end:
                layers[i] = None
            gc.collect()
    logger.info(f"Layers selected for quantization: {sorted(quant_layer_indices)}")

        
    layers[0] = layers[0].to(dev)
    print(layers[0])
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
  
    # args.deactive_amp = False args.epochs1=1
    if args.deactive_amp and args.epochs1>0:
        dtype = torch.float
        traincast = null_traincast
    else:
        dtype = args.dtype
        traincast = torch.amp.autocast
    
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.to('cpu')
            cache["i"] += 1
            # 由于seq_len一样，因此 attention_mask 和 position_embeddings 每次似乎是一样的，这里应该是有点冗余
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            
            raise ValueError

    
    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    # move embedding layer and first layer to cpu
    # print(cache["position_embeddings"] )
    #又把当前layer0给它还原回去了
    layers[0] = layers[0].module 
    layers[0] = layers[0].cpu() 
    
    

    if "llama" in args.net.lower() or "qwen" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise ValueError("Only support for qwen2.5, llama-2, Llama-3/3.1/3.2 now")
    torch.cuda.empty_cache()
    
    # same input of first layer for fp model and quant model
    
    inps = inps[:args.nsamples].to('cpu')
    quant_inps = inps
    # take output of fp model as input
    fp_outs = copy.deepcopy(inps)
    
        
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    
    position_embeddings = cache["position_embeddings"]
    

    #### Fuse parameters of RMSNorm and Rotation, abtain new model arch 
    if is_moe:
        logger.info("=== Skip norm fusion for MoE (router gate must stay unfused) ===")
    elif 'qwen3.' not in args.net.lower():
        logger.info(f"=== Start fuse nrom layers ===")
        fuse_indices = sorted(quant_layer_indices) if explicit_quant_layers else range(args.quant_end)
        for i in tqdm(fuse_indices):
            layer = layers[i].to(dev)
            
            for n,m in layer.named_modules():
                if 'input_layernorm' in n:
                    for name, module in  layer.named_modules():
                        if (isinstance(module, nn.Linear)) and( ('q_proj' in name) or ('k_proj' in name) or ('v_proj' in name)or ('in_proj_qkv' in name)or ('in_proj_z' in name) or ('in_proj_a' in name) or ('in_proj_b' in name)):
                            module.weight.data = module.weight.data * m.weight
                            
                    m.weight.data=torch.ones(m.weight.shape).to(dev).to(dtype)
                    
                if 'post_attention_layernorm' in n:
                    for name, module in  layer.named_modules():
                        if (isinstance(module, nn.Linear)) and( ('up_proj' in name) or ('gate_proj' in name)):
                            module.weight.data = module.weight.data * m.weight
                            
                    m.weight.data=torch.ones(m.weight.shape).to(dev).to(dtype)
            #if "llama" in args.net.lower() or "qwen" in args.net.lower():  
            #    qlayer = DecoderLayer(lm.model.config, layer, args) 
            #qlayer = qlayer.to(dev).to(dtype)
            qlayer = layer.to(dev).to(dtype)

            layers[i] = qlayer.to("cpu")
            del qlayer
            del layer
    

    
    fp_outs = fp_outs.to('cpu')
    quant_inps = quant_inps.to('cpu')

    ########### 
   
    for i in range(processing_end):
        #i=4
        should_quantize = i in quant_layer_indices
        logger.info(
            f"=== {'Start quantize' if should_quantize else 'Run FP prefix'} layer {i} ==="
        )
        qlayer = layers[i].to(dev)
        #if i==27:
        #    qlayer.to(float)
        if should_quantize:
            if is_moe:
                act_disturb, moe_stats = get_moe_act_means(qlayer, fp_outs, 32, 4,['q_proj', 'o_proj'],attention_mask=attention_mask,position_embeddings=position_embeddings)
            else:
                moe_stats = None
                if any(name.endswith('q_proj') for name, _ in qlayer.named_modules()):
                    act_disturb = get_act_means(qlayer, fp_outs, 8, 4,['q_proj', 'o_proj', 'up_proj', 'down_proj'],attention_mask=attention_mask,position_embeddings=position_embeddings)
                else:
                    act_disturb = get_act_means(qlayer, fp_outs, 8, 4,['in_proj_qkv', 'out_proj', 'up_proj', 'down_proj'],attention_mask=attention_mask,position_embeddings=position_embeddings)

        if should_quantize and args.auto_mix_precision:
            fp_inps = fp_outs.to('cpu')[:256].clone()
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=args.dtype):
                batch_size = args.batch_size * 2
                #args.batch_size = 2
                for j in tqdm(range(args.nsamples//batch_size)):
                    index = j * batch_size
                    fp_outs[index:index+batch_size,] = qlayer(fp_outs[index:index+batch_size,].to(dev), attention_mask=attention_mask,position_embeddings=position_embeddings).to('cpu').to(dtype)

        if not should_quantize:
            quant_inps.copy_(fp_outs)
            layers[i] = qlayer.to("cpu")
            del qlayer
            torch.cuda.empty_cache()
            continue

        logger.info(f"=== Prepared quantize layer {i} ===")
        for m in qlayer.modules():
            if type(m) == nn.Linear:
                m.weight.requires_grad_(False)
       
        if should_quantize:
            ################################
            #Stage0: prepare scale
            print("Doing scale Init...")
            with torch.no_grad():
                
                #qlayer.float() 
                print("Replacing")
                if args.auto_mix_precision:
                    replace_linear_with_TmpLinear_mix(qlayer, args, expc_list)
                elif is_moe:
                    replace_linear_with_TmpLinear_moe(qlayer, args, getattr(args, 'moe_num_groups', 1))
                else:
                    replace_linear_with_TmpLinear(qlayer, args)
                qlayer.float() 
                qlayer = qlayer.to(dev)
                if args.a1init:
                    print("Add scaling")
                    if any(name.endswith('q_proj') for name, _ in qlayer.named_modules()):
                        tmp = ((act_disturb['q_proj'].std(dim=0)/ act_disturb['q_proj'].std()).to(qlayer.self_attn.q_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.self_attn.q_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.self_attn.q_proj.a1.data = tmp
                        qlayer.self_attn.k_proj.a1.data = tmp
                        qlayer.self_attn.v_proj.a1.data = tmp

                        tmp = ((act_disturb['o_proj'].std(dim=0)/ act_disturb['o_proj'].std()).to(qlayer.self_attn.q_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.self_attn.o_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.self_attn.o_proj.a1.data = tmp
                    else:
                        tmp = ((act_disturb['in_proj_qkv'].std(dim=0)/ act_disturb['in_proj_qkv'].std()).to(qlayer.linear_attn.in_proj_qkv.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.linear_attn.in_proj_qkv.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.linear_attn.in_proj_qkv.a1.data = tmp
                        qlayer.linear_attn.in_proj_z.a1.data = tmp

                        tmp = ((act_disturb['out_proj'].std(dim=0)/ act_disturb['out_proj'].std()).to(qlayer.linear_attn.in_proj_qkv.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.linear_attn.out_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.linear_attn.out_proj.a1.data = tmp

                    
                    
                    if is_moe:
                        mlp = qlayer.mlp
                        num_experts = mlp.num_experts
                        group_size = mlp.moe_group_size
                        counts = moe_stats['counts']
                        for g in range(len(mlp.moe_shared)):
                            start = g * group_size
                            end = min((g + 1) * group_size, num_experts)
                            best_e = max(range(start, end), key=lambda e: counts[e].item())
                            mask = (moe_stats['all_sel'] == best_e).any(dim=1)
                            acts = moe_stats['all_in'][mask]
                            tmp = (acts.std(dim=0) / acts.std()).to(qlayer.self_attn.q_proj.a1.data)
                            tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                            tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                            if not torch.isfinite(tmp).all():
                                print("setting 1")
                                tmp.fill_(1.)
                            up_holder = mlp.moe_shared[g]['up_proj']
                            tmp = F.pad(tmp, (0, up_holder.expic - tmp.shape[-1]), mode="constant", value=1.)
                            tmp = tmp.reshape(up_holder.transdim1, up_holder.transdim2)
                            mlp.moe_shared[g]['gate_proj'].a1.data = tmp
                            mlp.moe_shared[g]['up_proj'].a1.data = tmp
                            mlp.moe_shared[g]['down_proj'].a1.data = torch.ones_like(mlp.moe_shared[g]['down_proj'].a1.data)
                    else:
                        tmp = ((act_disturb['up_proj'].std(dim=0)/ act_disturb['up_proj'].std()).to(qlayer.mlp.up_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.mlp.up_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.mlp.up_proj.a1.data  = tmp
                        qlayer.mlp.gate_proj.a1.data = tmp
                    
                        tmp = ((act_disturb['down_proj'].std(dim=0)/ act_disturb['down_proj'].std()).to(qlayer.mlp.up_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.mlp.down_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.mlp.down_proj.a1.data = tmp
                
                del act_disturb
                if is_moe:
                    del moe_stats
                print("Done scale Init...")
            
            ###############################################################  
            #Stage1: training transformation
            if args.nsamples1 == args.nsamples:
                args.nsamples1 = args.nsamples1 - args.nsamples//32
            qlayer = run_stage1_training(
                qlayer,
                args,
                i,
                quant_inps,
                fp_outs,
                attention_mask,
                attention_mask_batch,
                position_embeddings,
                traincast,
                dtype,
                dev,
                logger,
            )


        torch.cuda.empty_cache()
        ####
        ###############################################################                  
        # Stage2: finetuning all weights
        if args.finetuning_weights and should_quantize:
            qlayer = run_stage2_training(
                qlayer,
                args,
                i,
                expc_list if args.auto_mix_precision else None,
                quant_inps,
                fp_outs,
                attention_mask,
                attention_mask_batch,
                position_embeddings,
                traincast,
                dtype,
                dev,
                logger,
            )

        qlayer.to(dtype)
        with torch.no_grad():
            for n,m in qlayer.named_modules():
                if isinstance(m, FWTLinear):
                    m.pack_to_int8()

        if args.epochs1>0: 
            
            with torch.no_grad():
                #with torch.cuda.amp.autocast():
                with traincast(device_type='cuda',dtype=args.dtype):
                    batch_size = args.batch_size * 2
                    for j in tqdm(range(args.nsamples//batch_size)): 
                        index = j*batch_size
                        if explicit_quant_layers or i < args.quant_start or args.align <= 1:
                            quant_inps[index:index+batch_size,] = fp_outs[index:index+batch_size,]*1.
                        else:
                            quant_inps[index:index+batch_size,] = qlayer(quant_inps[index:index+batch_size,].to(dtype).to(dev), attention_mask=attention_mask,position_embeddings=position_embeddings).to('cpu')
                    
                # pack weight to int8
                
                layers[i] = qlayer.to("cpu")
            #if i==2:
            #    torch.save(fp_outs,"./layer2outputs.pth")
            print(fp_outs.flatten()[0:16])
            print(quant_inps.flatten()[0:16])
            logger.info(f"MSE: {(fp_outs[-args.nsamples//32:]- quant_inps[-args.nsamples//32:]).to(dev).to(torch.float32).pow(2).mean()}, Energy: {((fp_outs[:16]).to(dev).to(torch.float32).pow(2).mean())}")
            if args.align > 1:
                if i%args.align == args.align-1:
                    print('aligning')
                    for k in range(args.nsamples-args.nsamples//32):
                        quant_inps[k] = fp_outs[k]*1.
            
            del qlayer
        else:
            layers[i] = qlayer.to("cpu")
            if explicit_quant_layers:
                quant_inps.copy_(fp_outs)
        torch.cuda.empty_cache()
        if args.save_dir and args.save_per_layer and should_quantize:
            
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir,args.net, args.net+'+'+args.expc+'-layer'+str(i)+'.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(layers[i].state_dict(), save_path)
            print(f"Quantized model has been saved to：{save_path}")
        if args.save_per_layer:
            layers[i] = None 
            gc.collect()
        
    
    torch.cuda.empty_cache()
        
    del inps
    del quant_inps
    del fp_outs
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
