import math
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

import utils
from quantize.tmplinear import (
    FWTLinear,
    MoeSharedRotScale,
    TmpLinear,
    _is_moe_block,
    replace_TmpLinaer_with_FWTLinear,
    replace_TmpLinaer_with_FWTLinear_mix,
    replace_TmpLinaer_with_FWTLinear_moe,
    replace_linear_with_TmpLinear,
    replace_linear_with_TmpLinear_moe,
)
from quantize.utils import get_parameters


def init_training_ddp_if_needed(args):
    args.distributed_rank = 0
    args.distributed_world_size = 1
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not getattr(args, "quant_training_ddp", False):
        return
    if "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        raise ValueError("--quant_training_ddp requires launching with torchrun and WORLD_SIZE > 1")
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", args.local_rank))
    args.distributed_rank = dist.get_rank()
    args.distributed_world_size = dist.get_world_size()


def destroy_training_ddp_if_needed(args):
    if getattr(args, "quant_training_ddp", False) and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def null_traincast(device_type=None, dtype=None):
    return nullcontext()


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    return obj


def move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, tuple):
        return tuple(move_to_cpu(item) for item in obj)
    if isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]
    if isinstance(obj, dict):
        return {key: move_to_cpu(value) for key, value in obj.items()}
    return obj


def _get_n_set_parameters_byname(model, required_names):
    params = []
    for required_name in required_names:
        for name, param in model.named_parameters():
            if name.find(required_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return params


def _training_attention_mask(attention_mask, local_batch, args):
    if attention_mask is None:
        return None
    mask = attention_mask
    if mask.shape[0] != local_batch:
        mask = mask[:1].repeat(local_batch, 1, 1, 1)
    return mask if args.deactive_amp else mask.float()


def _prepare_stage1(qlayer, args):
    for _, param in qlayer.named_parameters():
        param.requires_grad = False

    wq_alpha = []
    scale_list0 = []
    scale_list1 = []
    scale_list2 = []
    w_list = []
    seen_a1 = set()
    seen_a2 = set()

    for _, module in qlayer.named_modules():
        if isinstance(module, TmpLinear):
            module.input_trans = True
            module.find_params()
            alpha = torch.zeros(
                module.quantizer.scale.shape,
                device=module.orilinear.weight.device,
                dtype=module.orilinear.weight.dtype,
            )
            if hasattr(module.quantizer, "alpha"):
                module.quantizer.alpha = nn.Parameter(alpha)
            else:
                module.quantizer.register_parameter("alpha", nn.Parameter(alpha))
            wq_alpha.append(module.quantizer.alpha)
            module.quantizer.alpha.requires_grad = True
            a1 = module._a1()
            a2 = module._a2()
            a2.requires_grad = True
            a1.requires_grad = True
            if id(a2) not in seen_a2:
                seen_a2.add(id(a2))
                scale_list1.append(a2)
            if id(a1) not in seen_a1:
                seen_a1.add(id(a1))
                scale_list2.append(a1)
            if module.orilinear.weight.requires_grad:
                w_list.append(module.orilinear.weight)

    scale_list0 += _get_n_set_parameters_byname(qlayer, ["Trans.linear"])
    lrscale0 = args.lscale_lr if args.transmask[0] == "1" else 0.0
    lrscale1 = args.lscale_lr if args.transmask[1] == "1" else 0.0
    lrscale2 = 2 * args.lscale_lr if args.transmask[2] == "1" else 0.0
    if args.transmask[3] == "1":
        param_groups = [
            {"params": wq_alpha, "lr": args.lwc_lr},
            {"params": scale_list0, "lr": lrscale0},
            {"params": scale_list1, "lr": lrscale1},
            {"params": scale_list2, "lr": lrscale2},
        ]
        if w_list:
            param_groups.insert(1, {"params": w_list, "lr": args.lw_lr})
    else:
        param_groups = [
            {"params": wq_alpha, "lr": args.lwc_lr},
            {"params": scale_list0, "lr": lrscale0},
            {"params": scale_list1, "lr": lrscale1},
            {"params": scale_list2, "lr": lrscale2},
        ]
        print(args.lwc_lr, lrscale0, lrscale1, lrscale2)
    return torch.optim.AdamW(param_groups, weight_decay=args.wd)


def _prepare_stage1_state_structure(qlayer, state_dict):
    for module_name, module in qlayer.named_modules():
        if not isinstance(module, TmpLinear):
            continue
        module.input_trans = True
        prefix = f"{module_name}." if module_name else ""
        for param_name in ("a1", "a2"):
            key = prefix + param_name
            if key in state_dict:
                setattr(module, param_name, nn.Parameter(torch.empty_like(state_dict[key])))
        for buffer_name in ("maxq", "scale", "zero"):
            key = prefix + "quantizer." + buffer_name
            if key in state_dict:
                setattr(module.quantizer, buffer_name, torch.empty_like(state_dict[key]))
        alpha_key = prefix + "quantizer.alpha"
        alpha = torch.empty_like(state_dict[alpha_key])
        if hasattr(module.quantizer, "alpha"):
            module.quantizer.alpha = nn.Parameter(alpha)
        else:
            module.quantizer.register_parameter("alpha", nn.Parameter(alpha))


def _build_stage1_schedulers(optimizer, epochs, steps_per_epoch):
    empty_optimizer_list = [
        torch.optim.AdamW([torch.tensor(0)], lr=optimizer.param_groups[k]["lr"])
        for k in range(len(optimizer.param_groups))
    ]
    return [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            empty_optimizer_list[k],
            T_max=epochs * steps_per_epoch,
            eta_min=optimizer.param_groups[k]["lr"] / 20,
        )
        for k in range(len(optimizer.param_groups))
    ]


def _refresh_tmp_weights(qlayer):
    for _, module in qlayer.named_modules():
        if isinstance(module, TmpLinear):
            module.quant_tmpweight()
            module.showflag = False


def _stage2_layer_groups():
    return [
        ["q_proj", "k_proj", "v_proj"],
        ["o_proj"],
        ["in_proj_qkv", "in_proj_z", "out_proj"],
        ["gate_proj", "up_proj"],
        ["down_proj"],
        ["ALL"],
    ]


def _set_fwt_params_by_name(module, required_names):
    params = []
    for name, param in module.named_parameters():
        if any(required_name in name for required_name in required_names):
            param.requires_grad = True
            params.append(param)
    return params


def _prepare_stage2(qlayer, args):
    for _, param in qlayer.named_parameters():
        param.requires_grad = False

    fwt_modules = [module for _, module in qlayer.named_modules() if isinstance(module, FWTLinear)]
    is_grouped = any(isinstance(module, MoeSharedRotScale) for module in qlayer.modules())
    is_moe = is_grouped or (
        hasattr(qlayer, "mlp")
        and hasattr(qlayer.mlp, "experts")
        and isinstance(qlayer.mlp.experts, nn.ModuleList)
    )
    if is_moe:
        weights = []
        for module in fwt_modules:
            module.weight.requires_grad = True
            weights.append(module.weight)
        weight_groups = [{"params": weights, "lr": args.lw_lr}]
    else:
        if hasattr(qlayer, "self_attn") and hasattr(qlayer.self_attn, "q_proj"):
            ordered_modules = [
                qlayer.self_attn.k_proj,
                qlayer.self_attn.v_proj,
                qlayer.self_attn.q_proj,
                qlayer.self_attn.o_proj,
                qlayer.mlp.up_proj,
                qlayer.mlp.gate_proj,
                qlayer.mlp.down_proj,
            ]
        elif hasattr(qlayer, "linear_attn"):
            ordered_modules = [
                qlayer.linear_attn.in_proj_qkv,
                qlayer.linear_attn.in_proj_z,
                qlayer.linear_attn.out_proj,
                qlayer.mlp.up_proj,
                qlayer.mlp.gate_proj,
                qlayer.mlp.down_proj,
            ]
        else:
            ordered_modules = fwt_modules
        weight_groups = []
        for module in ordered_modules:
            if not isinstance(module, FWTLinear):
                continue
            module.weight.requires_grad = True
            weight_groups.append(
                {
                    "params": [module.weight],
                    "lr": min(args.lw_lr, module.weight.std().item() / 50),
                }
            )

    scale_params = []
    linear_params = []
    a_params = []
    for _, module in qlayer.named_modules():
        if isinstance(module, TmpLinear):
            module.weight = module.weight.detach()
        if isinstance(module, FWTLinear):
            scale_params += _set_fwt_params_by_name(module, ["scale"])
            linear_params += _set_fwt_params_by_name(module, ["linear_"])
            a_params += _set_fwt_params_by_name(module, ["a1", "a2"])
        if isinstance(module, MoeSharedRotScale):
            a_params += _set_fwt_params_by_name(module, ["a1", "a2"])
            if module.Trans is not None:
                linear_params += _set_fwt_params_by_name(module.Trans, ["linear_"])

    param_groups = weight_groups + [
        {"params": scale_params, "lr": args.lw_lr / 5},
        {"params": linear_params, "lr": args.lt_lr},
        {"params": a_params, "lr": args.la_lr},
    ]
    param_groups = [group for group in param_groups if group["params"]]
    return torch.optim.AdamW(param_groups, weight_decay=args.wd)


def _build_stage2_schedulers(optimizer, total_steps):
    empty_optimizer_list = [
        torch.optim.AdamW([torch.tensor(0)], lr=optimizer.param_groups[k]["lr"])
        for k in range(len(optimizer.param_groups))
    ]
    return [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            empty_optimizer_list[k],
            T_max=total_steps,
            eta_min=optimizer.param_groups[k]["lr"] / 20,
        )
        for k in range(len(optimizer.param_groups))
    ]


def _replace_tmp_with_fwt_structure(model, args, layer_group, expc_list=None):
    for name, module in list(model.named_children()):
        if isinstance(module, TmpLinear):
            if not any(layer_name in name for layer_name in layer_group):
                continue
            expc = args.expc
            if expc_list is not None:
                if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                    expc = expc_list[0]
                elif "o_proj" in name:
                    expc = expc_list[1]
                elif "up_proj" in name or "gate_proj" in name:
                    expc = expc_list[2]
                elif "down_proj" in name:
                    expc = expc_list[3]
            fwt = FWTLinear()
            fwt.convert_form_tmplinear(
                module,
                bits=args.wbits,
                expc=expc,
                training_trans=args.training_trans,
                groupsize=args.groupsize,
                fast_nearest=args.fast_nearest,
            )
            fwt.bit_channel_convert(fast=True)
            setattr(model, name, fwt)
        else:
            _replace_tmp_with_fwt_structure(module, args, layer_group, expc_list)
    return model


def _convert_stage2_group(qlayer, args, layer_group, expc_list=None, structure_only=False, search_rank=None, search_world_size=None):
    if any(isinstance(m, MoeSharedRotScale) for m in qlayer.modules()):
        replace_TmpLinaer_with_FWTLinear_moe(qlayer, args, layer_group, expc_list, structure_only, search_rank, search_world_size)
        return qlayer
    if structure_only:
        return _replace_tmp_with_fwt_structure(qlayer, args, layer_group, expc_list)
    if args.auto_mix_precision:
        replace_TmpLinaer_with_FWTLinear_mix(qlayer, args, layer_group, expc_list)
    else:
        replace_TmpLinaer_with_FWTLinear(qlayer, args, layer_group)
    return qlayer


def _convert_all_stage2_groups(qlayer, args, expc_list=None, structure_only=False, search_rank=None, search_world_size=None):
    for layer_group in _stage2_layer_groups():
        qlayer = _convert_stage2_group(
            qlayer,
            args,
            layer_group,
            expc_list,
            structure_only=structure_only,
            search_rank=search_rank,
            search_world_size=search_world_size,
        )
    return qlayer


def _allgather_moe_search_results(qlayer, rank, world_size):
    """Redistribute sharded MoE lattice-search results across ranks.

    After a sharded Stage2 conversion, each rank holds the *real* searched
    weights only for the experts it owns (``e % world_size == rank``); the rest
    are placeholder null weights. Because the search is per-row and each rank
    searched a disjoint set of experts, an all_gather of the per-expert
    ``weight`` tensors (grouped by projection type so shapes match) recovers
    the full searched weights on every rank. ``scale`` is deterministic
    (``l2 * 2`` from the same pre-search weight), so it is identical across
    ranks and needs no gather.
    """
    if world_size <= 1:
        return

    proj_types = ["gate_proj", "up_proj", "down_proj"]
    for _, module in qlayer.named_modules():
        if not _is_moe_block(module):
            continue
        num_experts = len(module.experts)
        for p in proj_types:
            fwts = [getattr(module.experts[e], p) for e in range(num_experts)]
            if not all(isinstance(fwt, FWTLinear) for fwt in fwts):
                continue
            w = torch.stack([fwt.weight.data for fwt in fwts], dim=0)  # (E, oc, k)
            gathered = [torch.empty_like(w) for _ in range(world_size)]
            dist.all_gather(gathered, w)
            for e in range(num_experts):
                owner = e % world_size
                fwts[e].weight.data.copy_(gathered[owner][e])


def run_stage1_single(
    qlayer,
    args,
    quant_inps,
    fp_outs,
    attention_mask_batch,
    position_embeddings,
    traincast,
    dtype,
    dev,
    logger,
):
    optimizer = _prepare_stage1(qlayer, args)
    epochs = args.epochs1
    steps_per_epoch = args.nsamples1 // args.batch_size
    scheduler_list = _build_stage1_schedulers(optimizer, epochs, steps_per_epoch)
    loss_scaler = utils.NativeScalerWithGradNormCount()
    loss_func = torch.nn.MSELoss()
    with torch.no_grad():
        _refresh_tmp_weights(qlayer)

    for _ in range(epochs):
        loss_list = []
        norm_list = []
        for j in range(steps_per_epoch):
            index = j * args.batch_size
            with traincast(device_type="cuda", dtype=dtype):
                _refresh_tmp_weights(qlayer)
                quant_out = qlayer(
                    quant_inps[index:index + args.batch_size].to(dev),
                    attention_mask=attention_mask_batch,
                    position_embeddings=position_embeddings,
                )
                loss = loss_func(fp_outs[index:index + args.batch_size].to(dev), quant_out)

            if not math.isfinite(loss.item()):
                logger.info("Loss is NAN, stopping training")
                continue

            optimizer.zero_grad()
            loss_list.append(loss.detach().cpu())
            norm = loss_scaler(loss, optimizer, parameters=get_parameters(qlayer)).cpu()
            for k in range(len(optimizer.param_groups)):
                scheduler_list[k].step()
                optimizer.param_groups[k]["lr"] = scheduler_list[k].get_lr()[0]
            norm_list.append(norm.data)

            if j % 128 == 127:
                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(
                    f"batchs {j} loss:{loss_mean} norm:{norm_mean} "
                    f"max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} "
                )
                loss_list = []
                norm_list = []

    optimizer.zero_grad()
    del optimizer
    return qlayer


def run_stage1_ddp_loop(
    qlayer,
    args,
    attention_mask,
    position_embeddings,
    traincast,
    dtype,
    local_rank,
    rank,
    world_size,
    quant_inps=None,
    fp_outs=None,
    logger=None,
):
    if args.batch_size % world_size != 0:
        raise ValueError("--batch_size must be divisible by DDP world size for quant training")
    local_batch = args.batch_size // world_size
    dev = torch.device("cuda", local_rank)
    qlayer = qlayer.to(dev)
    optimizer = _prepare_stage1(qlayer, args)
    # MoE sparse routing means some experts receive no token in a step, so their
    # per-expert quantizer.alpha has no grad; allow unused params.
    qlayer = torch.nn.parallel.DistributedDataParallel(qlayer, device_ids=[local_rank], find_unused_parameters=True)
    epochs = args.epochs1
    steps_per_epoch = args.nsamples1 // args.batch_size
    scheduler_list = _build_stage1_schedulers(optimizer, epochs, steps_per_epoch)
    loss_scaler = utils.NativeScalerWithGradNormCount()
    loss_func = torch.nn.MSELoss()
    attention_mask = move_to_device(attention_mask, dev)
    position_embeddings = move_to_device(position_embeddings, dev)
    local_attention_mask = _training_attention_mask(attention_mask, local_batch, args)

    with torch.no_grad():
        _refresh_tmp_weights(qlayer.module)

    for _ in range(epochs):
        loss_list = []
        norm_list = []
        for j in range(steps_per_epoch):
            recv_quant_shape = (local_batch,) + tuple(quant_inps.shape[1:]) if rank == 0 else None
            recv_fp_shape = (local_batch,) + tuple(fp_outs.shape[1:]) if rank == 0 else None
            shape_obj = [(recv_quant_shape, recv_fp_shape)]
            dist.broadcast_object_list(shape_obj, src=0)
            recv_quant_shape, recv_fp_shape = shape_obj[0]
            local_quant = torch.empty(recv_quant_shape, dtype=dtype, device=dev)
            local_fp = torch.empty(recv_fp_shape, dtype=dtype, device=dev)

            if rank == 0:
                index = j * args.batch_size
                quant_chunks = list(quant_inps[index:index + args.batch_size].to(dev).chunk(world_size, dim=0))
                fp_chunks = list(fp_outs[index:index + args.batch_size].to(dev).chunk(world_size, dim=0))
            else:
                quant_chunks = None
                fp_chunks = None
            dist.scatter(local_quant, scatter_list=quant_chunks, src=0)
            dist.scatter(local_fp, scatter_list=fp_chunks, src=0)

            with traincast(device_type="cuda", dtype=dtype):
                _refresh_tmp_weights(qlayer.module)
                quant_out = qlayer(local_quant, attention_mask=local_attention_mask, position_embeddings=position_embeddings)
                loss = loss_func(local_fp, quant_out)

            finite_loss = torch.tensor(float(math.isfinite(loss.item())), device=dev)
            dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
            if finite_loss.item() == 0:
                if rank == 0 and logger is not None:
                    logger.info("Loss is NAN, stopping training")
                continue

            optimizer.zero_grad()
            norm = loss_scaler(loss, optimizer, parameters=get_parameters(qlayer.module)).detach()
            for k in range(len(optimizer.param_groups)):
                scheduler_list[k].step()
                optimizer.param_groups[k]["lr"] = scheduler_list[k].get_lr()[0]

            if rank == 0:
                loss_list.append(loss.detach().cpu())
                norm_list.append(norm.cpu())
                if j % 128 == 127:
                    loss_mean = torch.stack(loss_list).mean()
                    norm_mean = torch.stack(norm_list).mean()
                    if logger is not None:
                        logger.info(
                            f"batchs {j} loss:{loss_mean} norm:{norm_mean} "
                            f"max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} "
                        )
                    loss_list = []
                    norm_list = []

    optimizer.zero_grad()
    module = qlayer.module
    del optimizer, qlayer
    return module


def _run_stage2_train_loop(
    qlayer,
    args,
    quant_inps,
    fp_outs,
    attention_mask,
    position_embeddings,
    traincast,
    dtype,
    local_rank,
    rank,
    world_size,
    samplenums,
    epochs,
    logger=None,
):
    if args.batch_size % world_size != 0:
        raise ValueError("--batch_size must be divisible by DDP world size for quant training")
    local_batch = args.batch_size // world_size
    dev = torch.device("cuda", local_rank)
    qlayer = qlayer.to(dev)
    optimizer = _prepare_stage2(qlayer, args)
    # MoE sparse routing leaves some experts unused per step (their per-expert
    # weight/scale receive no grad); allow unused params.
    qlayer = torch.nn.parallel.DistributedDataParallel(qlayer, device_ids=[local_rank], find_unused_parameters=True)
    steps_per_epoch = samplenums // args.batch_size
    scheduler_list = _build_stage2_schedulers(optimizer, epochs * steps_per_epoch)
    loss_scaler = utils.NativeScalerWithGradNormCount()
    loss_func = torch.nn.MSELoss()
    attention_mask = move_to_device(attention_mask, dev)
    position_embeddings = move_to_device(position_embeddings, dev)
    local_attention_mask = _training_attention_mask(attention_mask, local_batch, args)

    for epoch in range(epochs):
        loss_list = []
        norm_list = []
        for j in range(steps_per_epoch):
            recv_quant_shape = (local_batch,) + tuple(quant_inps.shape[1:]) if rank == 0 else None
            recv_fp_shape = (local_batch,) + tuple(fp_outs.shape[1:]) if rank == 0 else None
            shape_obj = [(recv_quant_shape, recv_fp_shape)]
            dist.broadcast_object_list(shape_obj, src=0)
            recv_quant_shape, recv_fp_shape = shape_obj[0]
            local_quant = torch.empty(recv_quant_shape, dtype=dtype, device=dev)
            local_fp = torch.empty(recv_fp_shape, dtype=dtype, device=dev)

            if rank == 0:
                index = j * args.batch_size
                quant_chunks = list(quant_inps[index:index + args.batch_size].to(dev).chunk(world_size, dim=0))
                fp_chunks = list(fp_outs[index:index + args.batch_size].to(dev).chunk(world_size, dim=0))
            else:
                quant_chunks = None
                fp_chunks = None
            dist.scatter(local_quant, scatter_list=quant_chunks, src=0)
            dist.scatter(local_fp, scatter_list=fp_chunks, src=0)

            with traincast(device_type="cuda", dtype=dtype):
                quant_out = qlayer(local_quant, attention_mask=local_attention_mask, position_embeddings=position_embeddings)
                loss = loss_func(local_fp, quant_out)

            finite_loss = torch.tensor(float(math.isfinite(loss.item())), device=dev)
            dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
            if finite_loss.item() == 0:
                if rank == 0 and logger is not None:
                    logger.info("Loss is NAN, stopping training")
                continue

            optimizer.zero_grad()
            norm = loss_scaler(loss, optimizer, parameters=get_parameters(qlayer.module)).detach()
            for k in range(len(optimizer.param_groups)):
                scheduler_list[k].step()
                optimizer.param_groups[k]["lr"] = scheduler_list[k].get_lr()[0]
                if args.pvtuning:
                    if (epoch + j // 8) % 2 == 0:
                        if k > 0:
                            optimizer.param_groups[k]["lr"] = 0.0
                        else:
                            optimizer.param_groups[k]["lr"] = scheduler_list[k].get_lr()[0] * 10
                    elif k == 0:
                        optimizer.param_groups[k]["lr"] = 0.0

            if rank == 0:
                loss_list.append(loss.detach().cpu())
                norm_list.append(norm.cpu())
                if j % 128 == 127:
                    loss_mean = torch.stack(loss_list).mean()
                    if logger is not None:
                        logger.info(
                            f"batchs {j} loss:{loss_mean} lr:{optimizer.param_groups[0]['lr']} "
                            f"max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} "
                        )
                    loss_list = []
                    norm_list = []

    optimizer.zero_grad()
    module = qlayer.module
    del optimizer, qlayer
    return module


def run_stage2_single(
    qlayer,
    args,
    expc_list,
    quant_inps,
    fp_outs,
    attention_mask_batch,
    position_embeddings,
    traincast,
    dtype,
    dev,
    logger,
):
    if args.nsamples2 == args.nsamples:
        args.nsamples2 = args.nsamples2 - args.nsamples2 // 32
    for layer_group in _stage2_layer_groups():
        print("start finetuning all weights")
        qlayer = _convert_stage2_group(qlayer, args, layer_group, expc_list)
        qlayer = qlayer.to(dev)
        if layer_group == ["ALL"]:
            epochs = args.epochs2
            samplenums = args.nsamples2
        else:
            epochs = 0
            samplenums = 256
        if epochs == 0:
            continue

        optimizer = _prepare_stage2(qlayer, args)
        steps_per_epoch = samplenums // args.batch_size
        scheduler_list = _build_stage2_schedulers(optimizer, epochs * steps_per_epoch)
        loss_scaler = utils.NativeScalerWithGradNormCount()
        loss_func = torch.nn.MSELoss()
        for epoch in range(epochs):
            loss_list = []
            norm_list = []
            for j in range(steps_per_epoch):
                index = j * args.batch_size
                with traincast(device_type="cuda", dtype=dtype):
                    quant_out = qlayer(
                        quant_inps[index:index + args.batch_size].to(dev),
                        attention_mask=attention_mask_batch,
                        position_embeddings=position_embeddings,
                    )
                    loss = loss_func(fp_outs[index:index + args.batch_size].to(dev), quant_out)
                if not math.isfinite(loss.item()):
                    logger.info("Loss is NAN, stopping training")
                    continue

                optimizer.zero_grad()
                loss_list.append(loss.detach().cpu())
                norm = loss_scaler(loss, optimizer, parameters=get_parameters(qlayer)).cpu()
                for k in range(len(optimizer.param_groups)):
                    scheduler_list[k].step()
                    optimizer.param_groups[k]["lr"] = scheduler_list[k].get_lr()[0]
                    if args.pvtuning:
                        if (epoch + j // 8) % 2 == 0:
                            if k > 0:
                                optimizer.param_groups[k]["lr"] = 0.0
                            else:
                                optimizer.param_groups[k]["lr"] = scheduler_list[k].get_lr()[0] * 10
                        elif k == 0:
                            optimizer.param_groups[k]["lr"] = 0.0
                norm_list.append(norm.data)
                if j % 128 == 127:
                    loss_mean = torch.stack(loss_list).mean()
                    if logger is not None:
                        logger.info(f"batchs {j} loss:{loss_mean} lr:{optimizer.param_groups[0]['lr']} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} ")
                    loss_list = []
                    norm_list = []
        optimizer.zero_grad()
        del optimizer
    return qlayer


def run_stage2_ddp_loop(
    qlayer,
    args,
    expc_list,
    quant_inps,
    fp_outs,
    attention_mask,
    position_embeddings,
    traincast,
    dtype,
    local_rank,
    rank,
    world_size,
    logger=None,
):
    qlayer = qlayer.to(torch.device("cuda", local_rank))
    if rank == 0:
        print("start finetuning all weights")
    qlayer = _convert_all_stage2_groups(qlayer, args, expc_list)
    return _run_stage2_train_loop(
        qlayer,
        args,
        quant_inps,
        fp_outs,
        attention_mask,
        position_embeddings,
        traincast,
        dtype,
        local_rank,
        rank,
        world_size,
        args.nsamples2,
        args.epochs2,
        logger=logger,
    )


def broadcast_training_command(command):
    obj = [command]
    dist.broadcast_object_list(obj, src=0)


def run_stage1_training(
    qlayer,
    args,
    layer_idx,
    quant_inps,
    fp_outs,
    attention_mask,
    attention_mask_batch,
    position_embeddings,
    traincast,
    dtype,
    dev,
    logger,
):
    if not (getattr(args, "quant_training_ddp", False) and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return run_stage1_single(
            qlayer,
            args,
            quant_inps,
            fp_outs,
            attention_mask_batch,
            position_embeddings,
            traincast,
            dtype,
            dev,
            logger,
        )

    if args.batch_size % dist.get_world_size() != 0:
        raise ValueError("--batch_size must be divisible by DDP world size for --quant_training_ddp")
    qlayer = qlayer.to("cpu")
    torch.cuda.empty_cache()
    broadcast_training_command({"cmd": "stage1", "layer_idx": layer_idx, "nsamples1": args.nsamples1, "batch_size": args.batch_size})
    dist.broadcast_object_list([qlayer.state_dict()], src=0)
    dist.broadcast_object_list([(move_to_cpu(attention_mask), move_to_cpu(position_embeddings))], src=0)
    qlayer = run_stage1_ddp_loop(
        qlayer,
        args,
        attention_mask,
        position_embeddings,
        traincast,
        dtype,
        int(os.environ.get("LOCAL_RANK", 0)),
        dist.get_rank(),
        dist.get_world_size(),
        quant_inps=quant_inps,
        fp_outs=fp_outs,
        logger=logger,
    )
    return qlayer.to(dev)


def run_stage2_training(
    qlayer,
    args,
    layer_idx,
    expc_list,
    quant_inps,
    fp_outs,
    attention_mask,
    attention_mask_batch,
    position_embeddings,
    traincast,
    dtype,
    dev,
    logger,
):
    if not (getattr(args, "quant_training_ddp", False) and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return run_stage2_single(
            qlayer,
            args,
            expc_list,
            quant_inps,
            fp_outs,
            attention_mask_batch,
            position_embeddings,
            traincast,
            dtype,
            dev,
            logger,
        )

    if args.batch_size % dist.get_world_size() != 0:
        raise ValueError("--batch_size must be divisible by DDP world size for Stage2 DDP")
    if args.nsamples2 == args.nsamples:
        args.nsamples2 = args.nsamples2 - args.nsamples2 // 32
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    qlayer = qlayer.to("cpu")
    torch.cuda.empty_cache()
    broadcast_training_command({"cmd": "stage2", "layer_idx": layer_idx, "nsamples2": args.nsamples2, "batch_size": args.batch_size, "expc_list": expc_list})
    dist.broadcast_object_list([qlayer.state_dict()], src=0)
    dist.broadcast_object_list([(move_to_cpu(attention_mask), move_to_cpu(position_embeddings))], src=0)
    qlayer = qlayer.to(dev)
    print("start finetuning all weights")
    # Sharded lattice search: each rank searches its own subset of experts, then
    # all_gather redistributes the full searched weights. Removes the previous
    # rank0-only search + broadcast (which serialized search on a single GPU).
    qlayer = _convert_all_stage2_groups(qlayer, args, expc_list, search_rank=rank, search_world_size=world_size)
    _allgather_moe_search_results(qlayer, rank, world_size)
    qlayer = _run_stage2_train_loop(
        qlayer,
        args,
        quant_inps,
        fp_outs,
        attention_mask,
        position_embeddings,
        traincast,
        dtype,
        int(os.environ.get("LOCAL_RANK", 0)),
        dist.get_rank(),
        dist.get_world_size(),
        args.nsamples2,
        args.epochs2,
        logger=logger,
    )
    return qlayer.to(dev)


def _build_worker_qlayer(args, layer_idx):
    config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation)
    if (
        getattr(config, "model_type", "") == "qwen3_moe"
        or any("moe" in (arch or "").lower() for arch in (getattr(config, "architectures", None) or []))
    ):
        from models.qwen3_moe_per_expert import patch_qwen3_moe_per_expert
        patch_qwen3_moe_per_expert()
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=args.dtype,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
    qlayer = model.model.layers[layer_idx]
    is_moe = (
        getattr(config, "model_type", "") in ("qwen3_moe", "qwen2_moe")
        or any("moe" in (arch or "").lower() for arch in (getattr(config, "architectures", None) or []))
    )
    if is_moe:
        replace_linear_with_TmpLinear_moe(qlayer, args, getattr(args, "moe_num_groups", 1))
    else:
        replace_linear_with_TmpLinear(qlayer, args)
    del model
    return qlayer


def training_ddp_worker_loop(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if args.deactive_amp and args.epochs1 > 0:
        dtype = torch.float
        traincast = null_traincast
    else:
        dtype = args.dtype
        traincast = torch.amp.autocast

    while True:
        cmd_obj = [None]
        dist.broadcast_object_list(cmd_obj, src=0)
        cmd = cmd_obj[0]
        if cmd == "stop":
            break
        if not isinstance(cmd, dict) or cmd.get("cmd") not in {"stage1", "stage2"}:
            raise RuntimeError(f"rank {rank} received unexpected quant training command: {cmd}")

        args.batch_size = cmd["batch_size"]
        qlayer = _build_worker_qlayer(args, cmd["layer_idx"])
        state_obj = [None]
        dist.broadcast_object_list(state_obj, src=0)
        if cmd["cmd"] == "stage2":
            _prepare_stage1_state_structure(qlayer, state_obj[0])
        qlayer.load_state_dict(state_obj[0], assign=True, strict=True)
        context_obj = [None]
        dist.broadcast_object_list(context_obj, src=0)
        attention_mask, position_embeddings = context_obj[0]
        if cmd["cmd"] == "stage1":
            args.nsamples1 = cmd["nsamples1"]
            trained = run_stage1_ddp_loop(
                qlayer,
                args,
                attention_mask,
                position_embeddings,
                traincast,
                dtype,
                local_rank,
                rank,
                world_size,
            )
        else:
            args.nsamples2 = cmd["nsamples2"]
            qlayer = qlayer.to(torch.device("cuda", local_rank))
            qlayer = _convert_all_stage2_groups(
                qlayer,
                args,
                cmd.get("expc_list"),
                search_rank=rank,
                search_world_size=world_size,
            )
            _allgather_moe_search_results(qlayer, rank, world_size)
            trained = _run_stage2_train_loop(
                qlayer,
                args,
                None,
                None,
                attention_mask,
                position_embeddings,
                traincast,
                dtype,
                local_rank,
                rank,
                world_size,
                args.nsamples2,
                args.epochs2,
            )
        del trained
        del qlayer
        torch.cuda.empty_cache()


def stop_training_ddp_workers(args):
    if (
        getattr(args, "quant_training_ddp", False)
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_rank() == 0
        and dist.get_world_size() > 1
    ):
        broadcast_training_command("stop")
