import argparse
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from quantize.stage_training import (
    _convert_all_stage2_groups,
    _run_stage2_train_loop,
    move_to_cpu,
    run_stage1_ddp_loop,
    run_stage1_single,
    run_stage2_single,
)
from quantize.tmplinear import TmpLinear


class Stage1ToyLayer(nn.Module):
    def __init__(self, in_features, out_features, wbits, expc):
        super().__init__()
        linear = nn.Linear(in_features, out_features, bias=False)
        self.proj = TmpLinear(linear, wbits, expc=expc, training_trans=True)

    def prepare_stage1(self):
        for module in self.modules():
            if isinstance(module, TmpLinear):
                module.input_trans = True
                module.find_params()
                alpha = torch.zeros(
                    module.quantizer.scale.shape,
                    device=module.orilinear.weight.device,
                    dtype=module.orilinear.weight.dtype,
                )
                module.quantizer.register_parameter("alpha", nn.Parameter(alpha))

    def refresh_quant_weight(self):
        for module in self.modules():
            if isinstance(module, TmpLinear):
                module.quant_tmpweight()
                module.showflag = False

    def forward(self, x, **kwargs):
        self.refresh_quant_weight()
        return self.proj(x)


class Stage2ToyLayer(nn.Module):
    def __init__(self, features, wbits, expc):
        super().__init__()
        self.q_proj = TmpLinear(nn.Linear(features, features, bias=False), wbits, expc=expc, training_trans=True)

    def prepare_stage1_like_state(self):
        for module in self.modules():
            if isinstance(module, TmpLinear):
                module.input_trans = True
                module.find_params()
                alpha = torch.zeros(
                    module.quantizer.scale.shape,
                    device=module.orilinear.weight.device,
                    dtype=module.orilinear.weight.dtype,
                )
                module.quantizer.register_parameter("alpha", nn.Parameter(alpha))
                module.quant_tmpweight()

    def forward(self, x, **kwargs):
        return self.q_proj(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, device):
    set_seed(args.seed)
    model = Stage1ToyLayer(args.in_features, args.out_features, args.wbits, args.expc).to(device)
    model.prepare_stage1()
    return model


def build_optimizer(model, args):
    alpha_params = []
    a_params = []
    trans_params = []
    weight_params = []
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "quantizer.alpha" in name:
            alpha_params.append(param)
            param.requires_grad = True
        elif ".a1" in name or ".a2" in name:
            a_params.append(param)
            param.requires_grad = True
        elif "Trans.linear" in name:
            trans_params.append(param)
            param.requires_grad = True
        elif "orilinear.weight" in name:
            weight_params.append(param)
            param.requires_grad = True

    return torch.optim.AdamW(
        [
            {"params": alpha_params, "lr": args.lwc_lr},
            {"params": weight_params, "lr": args.lw_lr},
            {"params": trans_params, "lr": args.lscale_lr},
            {"params": a_params, "lr": args.la_lr},
        ],
        weight_decay=args.wd,
    )


def make_data(args, device):
    set_seed(args.seed + 1)
    x = torch.randn(args.steps * args.global_batch, args.in_features, device=device)
    teacher = torch.randn(args.in_features, args.out_features, device=device)
    y = x @ teacher
    return x, y


def train_single(args, x_cpu, y_cpu, device):
    model = build_model(args, device)
    optimizer = build_optimizer(model, args)
    loss_fn = nn.MSELoss()
    losses = []
    for step in range(args.steps):
        start = step * args.global_batch
        end = start + args.global_batch
        x = x_cpu[start:end].to(device)
        y = y_cpu[start:end].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu())
    return model.cpu().state_dict(), torch.stack(losses)


def train_ddp(args, rank, local_rank, world_size, x_cpu, y_cpu, device):
    model = build_model(args, device)
    ddp_model = DDP(model, device_ids=[local_rank], broadcast_buffers=True)
    optimizer = build_optimizer(ddp_model.module, args)
    loss_fn = nn.MSELoss()
    local_batch = args.global_batch // world_size
    losses = []

    for step in range(args.steps):
        start = step * args.global_batch + rank * local_batch
        end = start + local_batch
        x = x_cpu[start:end].to(device)
        y = y_cpu[start:end].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = ddp_model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())

    local_losses = torch.stack(losses)
    gathered_losses = [torch.empty_like(local_losses) for _ in range(world_size)]
    dist.all_gather(gathered_losses, local_losses)
    return ddp_model.module.cpu().state_dict(), torch.stack(gathered_losses).mean(dim=0).cpu()


def train_ddp_controller_flow(args, rank, local_rank, world_size, x_cpu, y_cpu, device):
    if rank == 0:
        set_seed(args.seed)
        source_model = Stage1ToyLayer(args.in_features, args.out_features, args.wbits, args.expc).cpu()
        for param in source_model.parameters():
            param.requires_grad = False
        state_obj = [source_model.state_dict()]
    else:
        state_obj = [None]
    torch.cuda.empty_cache()
    dist.broadcast_object_list(state_obj, src=0)
    set_seed(args.seed)
    model = Stage1ToyLayer(args.in_features, args.out_features, args.wbits, args.expc)
    model.load_state_dict(state_obj[0], assign=True, strict=True)
    if rank == 0:
        context_obj = [(move_to_cpu(None), move_to_cpu(None))]
    else:
        context_obj = [None]
    dist.broadcast_object_list(context_obj, src=0)
    attention_mask, position_embeddings = context_obj[0]

    if rank == 0:
        quant_inps = x_cpu
        fp_outs = y_cpu
    else:
        quant_inps = None
        fp_outs = None

    trained = run_stage1_ddp_loop(
        model,
        args,
        attention_mask,
        position_embeddings,
        lambda device_type, dtype: nullcontext(),
        torch.float32,
        local_rank,
        rank,
        world_size,
        quant_inps=quant_inps,
        fp_outs=fp_outs,
    )
    return trained.cpu().state_dict()


def train_single_controller_flow(args, x_cpu, y_cpu, device):
    set_seed(args.seed)
    model = Stage1ToyLayer(args.in_features, args.out_features, args.wbits, args.expc).to(device)
    for param in model.parameters():
        param.requires_grad = False
    trained = run_stage1_single(
        model,
        args,
        x_cpu,
        y_cpu,
        None,
        None,
        lambda device_type, dtype: nullcontext(),
        torch.float32,
        device,
        None,
    )
    return trained.cpu().state_dict(), torch.zeros(args.steps)


def train_single_stage2(args, x_cpu, y_cpu, device):
    set_seed(args.seed)
    source_model = Stage2ToyLayer(args.in_features, args.wbits, args.expc)
    source_model.prepare_stage1_like_state()
    source_state = source_model.state_dict()
    set_seed(args.seed)
    model = Stage2ToyLayer(args.in_features, args.wbits, args.expc)
    model.prepare_stage1_like_state()
    model.load_state_dict(source_state, assign=True, strict=True)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    trained = run_stage2_single(
        model,
        args,
        None,
        x_cpu,
        y_cpu,
        None,
        None,
        lambda device_type, dtype: nullcontext(),
        torch.float32,
        device,
        None,
    )
    with torch.no_grad():
        loss = nn.functional.mse_loss(trained(x_cpu.to(device)), y_cpu.to(device)).cpu()
    return trained.cpu().state_dict(), loss.unsqueeze(0)


def train_ddp_stage2(args, rank, local_rank, world_size, x_cpu, y_cpu, device):
    if rank == 0:
        set_seed(args.seed)
        source_model = Stage2ToyLayer(args.in_features, args.wbits, args.expc).cpu()
        source_model.prepare_stage1_like_state()
        for param in source_model.parameters():
            param.requires_grad = False
        state_obj = [source_model.state_dict()]
    else:
        state_obj = [None]
    dist.broadcast_object_list(state_obj, src=0)
    set_seed(args.seed)
    model = Stage2ToyLayer(args.in_features, args.wbits, args.expc)
    model.prepare_stage1_like_state()
    model.load_state_dict(state_obj[0], assign=True, strict=True)
    context_obj = [(move_to_cpu(None), move_to_cpu(None))] if rank == 0 else [None]
    dist.broadcast_object_list(context_obj, src=0)
    attention_mask, position_embeddings = context_obj[0]
    model = model.to(device)
    model = _convert_all_stage2_groups(
        model,
        args,
        None,
        structure_only=rank != 0,
    )
    converted_state = [model.cpu().state_dict()] if rank == 0 else [None]
    dist.broadcast_object_list(converted_state, src=0)
    model.load_state_dict(converted_state[0], assign=True, strict=True)
    trained = _run_stage2_train_loop(
        model,
        args,
        x_cpu if rank == 0 else None,
        y_cpu if rank == 0 else None,
        attention_mask,
        position_embeddings,
        lambda device_type, dtype: nullcontext(),
        torch.float32,
        local_rank,
        rank,
        world_size,
        args.nsamples2,
        args.epochs2,
    )
    local_batch = args.global_batch // world_size
    start = rank * local_batch
    end = start + local_batch
    with torch.no_grad():
        local_loss = nn.functional.mse_loss(
            trained(x_cpu[start:end].to(device)),
            y_cpu[start:end].to(device),
        )
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    local_loss /= world_size
    return trained.cpu().state_dict(), local_loss.cpu().unsqueeze(0)


def compare_state_dicts(single_state, ddp_state):
    max_abs = 0.0
    worst_key = None
    compared = 0
    for key, single_value in single_state.items():
        if key not in ddp_state or not torch.is_floating_point(single_value):
            continue
        ddp_value = ddp_state[key].to(single_value.dtype)
        diff = (single_value - ddp_value).abs().max().item()
        compared += 1
        if diff > max_abs:
            max_abs = diff
            worst_key = key
    return compared, worst_key, max_abs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_features", type=int, default=64)
    parser.add_argument("--out_features", type=int, default=32)
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--expc", type=str, default="24to8")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--global_batch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--lw_lr", type=float, default=2e-5)
    parser.add_argument("--lscale_lr", type=float, default=5e-3)
    parser.add_argument("--la_lr", type=float, default=2e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=5e-5)
    parser.add_argument("--controller_flow", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    args = parser.parse_args()
    if args.stage2:
        args.out_features = args.in_features
    args.batch_size = args.global_batch
    args.nsamples1 = args.steps * args.global_batch
    args.epochs1 = 1
    args.epochs2 = 1
    args.nsamples2 = args.steps * args.global_batch
    args.nsamples = max(args.nsamples1, args.nsamples2)
    args.transmask = "1111"
    args.deactive_amp = True
    args.dtype = torch.float32
    args.auto_mix_precision = False
    args.training_trans = True
    args.groupsize = -1
    args.fast_nearest = True
    args.pvtuning = False
    args.lt_lr = args.lscale_lr

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.global_batch % world_size != 0:
        raise ValueError("global_batch must be divisible by world_size for this validation")

    set_seed(args.seed + 1)
    x_cpu = torch.randn(args.steps * args.global_batch, args.in_features)
    teacher = torch.randn(args.in_features, args.out_features)
    y_cpu = x_cpu @ teacher

    if rank == 0:
        if args.stage2:
            single_state, single_losses = train_single_stage2(args, x_cpu, y_cpu, device)
        elif args.controller_flow:
            single_state, single_losses = train_single_controller_flow(args, x_cpu, y_cpu, device)
        else:
            single_state, single_losses = train_single(args, x_cpu, y_cpu, device)
    else:
        single_state = None
        single_losses = None

    dist.barrier()
    if args.stage2:
        ddp_state, ddp_losses = train_ddp_stage2(args, rank, local_rank, world_size, x_cpu, y_cpu, device)
    elif args.controller_flow:
        ddp_state = train_ddp_controller_flow(args, rank, local_rank, world_size, x_cpu, y_cpu, device)
        ddp_losses = single_losses if rank == 0 else None
    else:
        ddp_state, ddp_losses = train_ddp(args, rank, local_rank, world_size, x_cpu, y_cpu, device)

    if rank == 0:
        compared, worst_key, max_abs = compare_state_dicts(single_state, ddp_state)
        loss_diff = (single_losses - ddp_losses).abs().max().item()
        print(f"world_size={world_size} global_batch={args.global_batch} local_batch={args.global_batch // world_size}")
        print(f"controller_flow={args.controller_flow}")
        print(f"stage2={args.stage2}")
        print(f"single_losses={single_losses.tolist()}")
        print(f"ddp_mean_local_losses={ddp_losses.tolist()}")
        print(f"loss_max_abs_diff={loss_diff:.8g}")
        print(f"state_compared={compared} state_max_abs_diff={max_abs:.8g} worst_key={worst_key}")
        if max_abs > args.atol:
            raise SystemExit(f"DDP validation failed: state diff {max_abs} > {args.atol}")
        print("DDP validation passed")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
