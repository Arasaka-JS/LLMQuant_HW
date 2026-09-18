#!/usr/bin/env python3
"""Self-check for the vectorized MoE expert forward.

Checks ``LiftQuant/models/moe_fast_eval.py`` against a straightforward per-expert
reference loop.  The correctness of this optimisation is *device independent*, so
the whole check runs on CPU by default -- no CUDA/NPU card is needed.

Covered:
  * routing shapes: decode (M=1), short, prefill, multi-batch
  * pathological routing: one expert owning every token, experts with no tokens
  * dtypes: float32 (tight tolerance => proves algorithmic equivalence) and bfloat16
  * both bucket strategies: sync-free (tmax = num_slots) and exact (max(counts))
  * block conversion: in-place replacement + release of the per-expert fp cache

Usage:
    python allq/tools/moe_fast_selfcheck.py            # CPU, fast
    python allq/tools/moe_fast_selfcheck.py --device cuda
    python allq/tools/moe_fast_selfcheck.py --experts 64 --hidden 512   # bigger

Exit code 0 means every check passed.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFTQUANT_ROOT = PROJECT_ROOT / "LiftQuant"
sys.path.insert(0, str(LIFTQUANT_ROOT))

from models.moe_fast_eval import (  # noqa: E402
    FastMoEBlock,
    FastMoEExperts,
    convert_moe_blocks_to_fast,
    stack_experts_from_block,
)
from quantize.tmplinear import FWTLinear  # noqa: E402


FP32_RTOL = 1e-5
BF16_RTOL = 1e-2


class MLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = F.silu


class FakeMoEBlock(nn.Module):
    """Minimal stand-in for a per-expert MoE block (gate + ModuleList experts)."""

    def __init__(self, num_experts, hidden, inter):
        super().__init__()
        self.experts = nn.ModuleList([MLP(hidden, inter) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden, num_experts, bias=False)


def reference_loop(block, hidden, idx, weights):
    """Per-expert Python loop -- the semantics the fast path must reproduce."""
    out = torch.zeros_like(hidden)
    for e in range(len(block.experts)):
        tok, pos = (idx == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        expert = block.experts[e]
        x = hidden[tok]
        y = expert.down_proj(expert.act_fn(expert.gate_proj(x)) * expert.up_proj(x))
        out.index_add_(0, tok, y * weights[tok, pos].unsqueeze(-1))
    return out


def routing(num_tokens, num_experts, top_k, mode, device):
    """Routing indices for a token batch.

    ``mode="random"`` reproduces what a real ``top-k`` router produces: ``k``
    **distinct** experts per token (``torch.topk`` never repeats an index).  That
    matters because the sync-free bucket relies on ``counts[e] <= num_tokens``,
    which only holds for distinct top-k.  ``mode="duplicate"`` deliberately
    samples with replacement to exercise ``assume_distinct_topk=False``.
    """
    if mode == "single":  # every token routed to expert 0 (only valid without the bound)
        idx = torch.zeros(num_tokens, top_k, dtype=torch.long, device=device)
    elif mode == "spread":  # deterministic round-robin
        idx = (torch.arange(num_tokens * top_k, device=device) % num_experts).reshape(num_tokens, top_k)
    elif mode == "duplicate":  # with replacement -> an expert may own several slots of one token
        idx = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
    else:  # "random" (default): distinct experts per token, like torch.topk
        logits = torch.rand(num_tokens, num_experts, device=device)
        idx = torch.topk(logits, min(top_k, num_experts), dim=-1).indices
    weights = torch.rand(num_tokens, top_k, device=device)
    return idx, weights


class RouterStub(nn.Module):
    """Stand-in router returning the native (logits, weights, indices) triple."""

    def __init__(self, hidden, num_experts, top_k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden))
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        logits = F.linear(x, self.weight)
        probs = torch.softmax(logits.float(), dim=-1)
        vals, idx = torch.topk(probs, self.top_k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True)
        return logits, vals.to(logits.dtype), idx


def _rel(diff, ref):
    denom = ref.abs().mean().item()
    return diff.mean().item() / denom if denom else diff.mean().item()


def check_numerics(block, fast, device, dtype, cases, results):
    for desc, num_tokens, top_k, mode in cases:
        idx, weights = routing(num_tokens, len(block.experts), top_k, mode, device)
        # The real router emits routing weights in the model dtype; match that so
        # the reference loop's index_add_ sees a single dtype.
        weights = weights.to(dtype)
        hidden = torch.randn(num_tokens, block.experts[0].gate_proj.in_features,
                             device=device, dtype=dtype)
        with torch.no_grad():
            ref = reference_loop(block, hidden, idx, weights)
            got = fast(hidden, idx, weights)
        diff = (ref.float() - got.float()).abs()
        rel = _rel(diff, ref.float())
        tol = FP32_RTOL if dtype == torch.float32 else BF16_RTOL
        results.append((rel <= tol,
                        f"[{str(dtype).split('.')[-1]:<8}] {desc:<26}"
                        f"max|d|={diff.max().item():.3e} mean_rel={rel:.3e} (tol {tol:g})"))


def check_block_forward(block, fast_experts, device, dtype, results):
    """FastMoEBlock must match the same gate+loop composition for (B, T, H)."""
    hidden_dim = block.experts[0].gate_proj.in_features
    router = RouterStub(hidden_dim, len(block.experts), 4).to(device, dtype)
    fast_block = FastMoEBlock(gate=router, experts=fast_experts).to(device, dtype).eval()
    hidden = torch.randn(2, 16, hidden_dim, device=device, dtype=dtype)
    with torch.no_grad():
        flat = hidden.view(-1, hidden_dim)
        _, weights, idx = router(flat)
        ref = reference_loop(block, flat, idx, weights).view_as(hidden)
        got = fast_block(hidden)
    diff = (ref.float() - got.float()).abs()
    rel = _rel(diff, ref.float())
    tol = FP32_RTOL if dtype == torch.float32 else BF16_RTOL
    results.append((rel <= tol,
                    f"[{str(dtype).split('.')[-1]:<8}] FastMoEBlock (2,16,H)      "
                    f"max|d|={diff.max().item():.3e} mean_rel={rel:.3e}"))


def check_strategies(block, device, results):
    """Sync-free bucket and exact bucket must agree, for both bound flavours."""
    hidden_dim = block.experts[0].gate_proj.in_features
    num_experts = len(block.experts)
    cases = (
        # (label, tokens, top_k, routing mode, assume_distinct_topk)
        ("decode", 1, 4, "random", True),
        ("prefill", 128, 4, "random", True),
        ("decode/dup", 1, 4, "duplicate", False),
        ("prefill/dup", 96, 4, "duplicate", False),
    )
    for desc, num_tokens, top_k, mode, distinct in cases:
        sync_free = stack_experts_from_block(
            block, dtype=torch.float32, sync_free_row_budget=10 ** 9,
            assume_distinct_topk=distinct,
        )
        exact = stack_experts_from_block(block, dtype=torch.float32, sync_free_row_budget=0)
        idx, weights = routing(num_tokens, num_experts, top_k, mode, device)
        hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float32)
        with torch.no_grad():
            a = sync_free.to(device)(hidden, idx, weights)
            b = exact.to(device)(hidden, idx, weights)
        diff = (a - b).abs()
        rel = _rel(diff, b)
        results.append((rel <= 1e-6,
                        f"[strategy] {desc:<12} sync-free vs exact   "
                        f"max|d|={diff.max().item():.3e} mean_rel={rel:.3e}"))


def check_conversion(results):
    """convert_moe_blocks_to_fast must replace the block and release fp caches."""
    num_experts, hidden, inter = 4, 32, 64
    holder = nn.Sequential()
    for _ in range(2):
        block = FakeMoEBlock(num_experts, hidden, inter)
        for expert in block.experts:
            for name, shape in (("gate_proj", (inter, hidden)), ("up_proj", (inter, hidden))):
                lin = FWTLinear()
                lin.oc, lin.ic = shape
                lin.register_buffer("_weight_fp", torch.randn(*shape), persistent=False)
                setattr(expert, name, lin)
            lin = FWTLinear()
            lin.oc, lin.ic = hidden, inter
            lin.register_buffer("_weight_fp", torch.randn(hidden, inter), persistent=False)
            lin.register_buffer("packed_weight", torch.zeros(8, dtype=torch.uint8), persistent=False)
            expert.down_proj = lin
        holder.append(block)
    converted = convert_moe_blocks_to_fast(holder)
    leaked = sum(
        1 for m in holder.modules()
        if isinstance(m, FWTLinear) and ("_weight_fp" in m._buffers or "packed_weight" in m._buffers)
    )
    fused = [m for m in holder.modules() if isinstance(m, FastMoEExperts)]
    ok = len(converted) == 2 and leaked == 0 and len(fused) == 2
    results.append((ok, f"[convert]  blocks={len(converted)} fused={len(fused)} "
                        f"leaked_fp_cache={leaked} names={converted}"))


def check_fallback(block, device, results):
    """The padded-row cap must switch to the per-expert loop and stay correct."""
    hidden_dim = block.experts[0].gate_proj.in_features
    for dtype in (torch.float32, torch.bfloat16):
        block_d = block.to(dtype)
        # max_padded_rows=1 forces the guard-rail path for every call.
        fast = stack_experts_from_block(
            block_d, dtype=dtype, max_padded_rows=1
        ).to(device).eval()
        for desc, num_tokens, top_k, mode in (("decode M=1", 1, 4, "random"),
                                              ("M=32 spread", 32, 2, "spread")):
            idx, weights = routing(num_tokens, len(block.experts), top_k, mode, device)
            weights = weights.to(dtype)
            hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
            with torch.no_grad():
                ref = reference_loop(block_d, hidden, idx, weights)
                got = fast(hidden, idx, weights)
            diff = (ref.float() - got.float()).abs()
            rel = _rel(diff, ref.float())
            tol = FP32_RTOL if dtype == torch.float32 else BF16_RTOL
            results.append((rel <= tol,
                            f"[fallback] {desc:<16} {str(dtype).split('.')[-1]:<9}"
                            f"max|d|={diff.max().item():.3e} mean_rel={rel:.3e} (tol {tol:g})"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--inter", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    block = FakeMoEBlock(args.experts, args.hidden, args.inter).to(device)
    results = []

    cases = [
        ("decode M=1", 1, 4, "random"),
        ("short M=3", 3, 4, "random"),
        ("prefill M=64", 64, 4, "random"),
        ("batch M=2x8", 16, 4, "random"),
        ("single-expert routing", 12, 4, "single"),
        ("one slot per expert", args.experts, 1, "spread"),
    ]
    for dtype in (torch.float32, torch.bfloat16):
        # Same weights in both dtypes so the reference and the fast path see
        # identical values; the dtype cast keeps the reference loop consistent.
        block_d = block.to(dtype)
        fast = stack_experts_from_block(block_d, dtype=dtype).to(device).eval()
        check_numerics(block_d, fast, device, dtype, cases, results)
        check_block_forward(block_d, fast, device, dtype, results)

    check_strategies(block, device, results)
    check_fallback(block, device, results)
    check_conversion(results)

    print(f"moe_fast_eval self-check  (device={device}, experts={args.experts}, "
          f"hidden={args.hidden}, inter={args.inter})")
    print("=" * 80)
    for ok, line in results:
        print(f"{'PASS' if ok else 'FAIL'}  {line}")
    failed = sum(1 for ok, _ in results if not ok)
    print("=" * 80)
    print(f"{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
