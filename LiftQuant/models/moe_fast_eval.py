"""Vectorized, sync-free MoE expert forward for quantized LiftQuant evaluation.

Why this exists
---------------
Both HF's native ``Qwen3MoeExperts.forward`` (fused 3D weights) and LiftQuant's
``PerExpertQwen3MoeSparseMoeBlock.forward`` (per-expert ``ModuleList``) run a
*host-side Python loop* over the routed experts, and the loop body forces
device->host synchronisations (``.nonzero()``, ``torch.where``, ``int()``).

For Qwen3-30B-A3B (48 layers, 128 experts, top-8) that is ~512 Python-level
module calls and 100+ syncs per MoE layer per forward pass, which dwarfs the
actual GEMM work.  Measured on an A800 (SM80) with real layer-0 weights:

    prefill (128 tokens/expert)  loop 10.19 ms/layer  ->  bmm 1.07 ms/layer  (9.5x)
    decode  (1 token, top-8)     loop  0.857 ms/layer ->  bmm 0.276 ms/layer (3.1x)
    host sync alone              0.175 ms/layer (~8.4 ms per forward over 48 layers)

This module replaces the loop with a sort/bucket + batched-GEMM formulation that
runs entirely on the GPU:

    1. group the (token, expert) routing slots by expert (GPU stable argsort),
    2. pad each expert's slots to ``Tmax`` rows -> (E, Tmax, H),
    3. one ``torch.bmm`` for the fused gate+up projection  (E, Tmax, 2I),
    4. ``act(gate) * up`` then one ``torch.bmm`` for down   (E, Tmax, H),
    5. gather the real rows back, scale by the routing weights and
       ``index_add_`` into the output.

Per layer that is ~8 kernel launches instead of 384 (prefill) / 24 (decode), and
no synchronisation.  ``torch._grouped_mm`` (which would be even better) requires
compute capability 9.0 and is unavailable here, so ``torch.bmm`` is used.

Only the evaluation/inference path uses this; the quantization path
(``stage_training.py``) keeps the per-expert structure it needs for per-expert
learning rates and weight fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize.tmplinear import FWTLinear


__all__ = [
    "FastMoEExperts",
    "FastMoEBlock",
    "stack_experts_from_block",
    "convert_moe_block_to_fast",
    "convert_moe_blocks_to_fast",
    "SYNC_FREE_ROW_BUDGET",
    "SYNC_FREE_MAX_SLOTS",
    "MAX_PADDED_ROWS",
    "ASSUME_DISTINCT_TOPK",
]

# Padded-row budget below which the data-independent bucket is used instead of
# the exact `counts.max()` (which costs one host sync per layer).  With
# ``assume_distinct_topk`` the bucket bound is ``num_tokens``, so the condition
# ``bound * num_experts <= budget`` is really a cap on the token count.
#
# Measured on an A800 (bf16, real layer-0 weights, 128 experts, top-8):
#   * M=1   (decode): bound=1 -> sync-free is bit-identical to exact, 1.02x faster
#   * M=8   : bound=8 -> 1.00x (the padding costs what the sync saves)
#   * M=128 : bound=128 would roughly double the layer time, hence the cap below
# Only very short sequences profit, so the default stays 0 (always exact).  A
# backend with much more expensive syncs (Ascend) can opt in with a modest
# budget, e.g. 4096 rows (= M <= 32 for 128 experts).
SYNC_FREE_ROW_BUDGET = 0

# Hard cap for the sync-free bucket bound regardless of the configured budget, so
# a runaway budget can never size the padded tensor by a huge bound on a long
# prefill (num_experts * 16384 rows would be ~8.6 GB and ~22x slower).  The bound
# is `num_tokens` when ``ASSUME_DISTINCT_TOPK`` holds, else `num_slots`.
SYNC_FREE_MAX_SLOTS = 256

# Cap on the padded (num_experts * tmax) rows.  With real routing the bucket is
# tiny (measured ~157 rows for a 2048-token prefill), but a pathological routing
# where one expert owns every slot would pad *all* experts to that count.  Above
# this cap the forward falls back to a per-expert loop over the fused tensors,
# which is correct but slower -- a guard rail, not the common path.
MAX_PADDED_ROWS = 1 << 16

# Real top-k routers (``Qwen3MoeTopKRouter`` uses ``torch.topk``) return k
# *distinct* experts per token, so a single expert receives at most one slot per
# token -> ``counts[e] <= num_tokens``.  That is a k-times tighter (and still
# provable) bucket than ``num_slots``, which is what makes the sync-free bucket
# cheap for decode (bound = 1).  Set to False for a router that may repeat an
# expert id within one token; the safe ``num_slots`` bound is then used.
ASSUME_DISTINCT_TOPK = True


def _module_weight(module):
    """Return the (already dequantized) fp weight of a quantized or plain linear."""
    if isinstance(module, FWTLinear):
        if not hasattr(module, "_weight_fp"):
            module.materialize()
        return module._weight_fp
    return module.weight


class FastMoEExperts(nn.Module):
    """Fused expert weights ``(E, 2I, H)`` / ``(E, H, I)`` with a vectorized forward.

    The buffer layout matches HF's native ``Qwen3MoeExperts``
    (``gate_up_proj`` / ``down_proj``), so this is a drop-in replacement for the
    expert container of a MoE block.
    """

    def __init__(self, gate_up, down, act_fn, gate_up_bias=None, down_bias=None,
                 sync_free_row_budget=0, max_padded_rows=MAX_PADDED_ROWS,
                 assume_distinct_topk=True):
        super().__init__()
        num_experts, two_i, hidden = gate_up.shape
        if two_i % 2:
            raise ValueError(f"fused gate_up projection must have an even dim, got {two_i}")
        self.num_experts = num_experts
        self.intermediate_dim = two_i // 2
        self.hidden_dim = hidden
        self.act_fn = act_fn
        # Maximum number of padded rows (num_experts * tmax) we are willing to
        # spend in order to avoid the one host sync per layer (see _select_tmax).
        # 0 keeps the default "exact bucket" behaviour (measured fastest on CUDA).
        self.sync_free_row_budget = int(sync_free_row_budget)
        # Above this many padded rows the forward switches to the per-expert
        # fallback so pathological routing cannot blow up memory.
        self.max_padded_rows = int(max_padded_rows)
        # top-k routers return distinct experts per token, which lets the
        # sync-free bucket use `num_tokens` instead of `num_slots` (k times less
        # padding).  Set False for routers that may repeat an expert.
        self.assume_distinct_topk = bool(assume_distinct_topk)
        # Non-persistent: the eval-time cache must never leak into a checkpoint.
        self.register_buffer("_gate_up", gate_up.contiguous(), persistent=False)
        self.register_buffer("_down", down.contiguous(), persistent=False)
        if gate_up_bias is not None:
            self.register_buffer("_gate_up_bias", gate_up_bias.contiguous(), persistent=False)
        else:
            self._gate_up_bias = None
        if down_bias is not None:
            self.register_buffer("_down_bias", down_bias.contiguous(), persistent=False)
        else:
            self._down_bias = None

    def _select_tmax(self, counts, num_slots, num_tokens, num_experts):
        """Size of the padded per-expert bucket, avoiding a host sync when enabled.

        A *provably* sufficient bucket for **any** routing distribution is
        ``tmax = num_slots`` (a single expert could own every slot).  For the
        top-k routers used everywhere in this repo a tighter, equally provable
        bound exists: ``torch.topk`` returns ``k`` *distinct* experts per token,
        so a single expert receives at most one slot per token and therefore
        ``counts[e] <= num_tokens``.  That is ``k`` times less padding.

        Padding to ``tmax`` costs ``num_experts * tmax`` rows, which is
        negligible for decode and prohibitive for long prefills, so the sync-free
        branch is only taken while

            ``bounds <= SYNC_FREE_MAX_SLOTS`` and
            ``bounds * num_experts <= self.sync_free_row_budget``.

        ``sync_free_row_budget = 0`` (the default) always uses the exact bucket
        ``max(counts)``, which costs one host sync per layer.  Measured on an
        A800 (bf16, real layer-0 weights): the sync itself is only ~0.04 ms,
        while padding decode up to ``tmax = num_slots`` costs 0-13% more, so
        exact is the better default on CUDA.  A backend where a host sync is much
        more expensive (Ascend) or that forbids ``.item()`` (CUDA graphs) can opt
        in by raising the budget.

        If ``assume_distinct_topk`` is false the safe ``num_slots`` bound is used
        instead.  Note the failure mode of an under-sized bucket is loud, not
        silent: the scatter ``x_sorted[sel_e, pos]`` raises ``IndexError``.

        Note also that a "capacity factor + overflow rounds" scheme cannot be
        both sync-free and provably lossless: covering every distribution
        requires ``rounds * capacity >= num_slots``, i.e. ``num_experts *
        num_slots`` padded rows in total -- the same worst case as
        ``tmax = num_slots``.
        """
        if num_slots == 0:
            return 1
        if self.sync_free_row_budget > 0:
            bound = num_tokens if self.assume_distinct_topk else num_slots
            if (bound <= SYNC_FREE_MAX_SLOTS
                    and bound * num_experts <= self.sync_free_row_budget):
                return bound
        return int(counts.max().item())

    @torch.no_grad()
    def _forward_expert_loop(self, hidden_states, tok, sel_e, slot_weights, counts):
        """Guard-rail fallback: per-expert loop over the fused tensors.

        Used when the padded bucket would exceed ``max_padded_rows`` (pathological
        routing).  Correct but slow -- it exists so evaluation degrades instead of
        running out of memory.
        """
        I = self.intermediate_dim
        final = hidden_states.new_zeros(hidden_states.shape)
        for e in torch.nonzero(counts, as_tuple=False).flatten().tolist():
            rows = torch.nonzero(sel_e == e, as_tuple=False).flatten()
            rows_tok = tok.index_select(0, rows)
            x = hidden_states.index_select(0, rows_tok)
            gate_up = F.linear(x, self._gate_up[e])
            if self._gate_up_bias is not None:
                gate_up = gate_up + self._gate_up_bias[e]
            y = F.linear(self.act_fn(gate_up[:, :I]) * gate_up[:, I:], self._down[e])
            if self._down_bias is not None:
                y = y + self._down_bias[e]
            y = y * slot_weights.index_select(0, rows).unsqueeze(-1)
            final.index_add_(0, rows_tok, y)
        return final

    @torch.no_grad()
    def forward(self, hidden_states, top_k_index, top_k_weights):
        """hidden_states: (M, H); top_k_index: (M, k); top_k_weights: (M, k)."""
        M, hidden = hidden_states.shape
        E = self.num_experts
        I = self.intermediate_dim
        device = hidden_states.device
        k = top_k_index.shape[1]

        flat_e = top_k_index.reshape(-1).to(torch.long)
        num_slots = flat_e.numel()
        slot_tok = torch.arange(M, device=device, dtype=torch.long).repeat_interleave(k)

        # Group the routing slots by expert (stable -> keeps the original order
        # inside an expert, which keeps the result reproducible).
        order = torch.argsort(flat_e, stable=True)
        sel_e = flat_e[order]
        tok = slot_tok[order]
        slot_weights = top_k_weights.reshape(-1)[order]

        counts = torch.bincount(sel_e, minlength=E)
        tmax = self._select_tmax(counts, num_slots, M, E)
        if tmax * E > self.max_padded_rows:
            return self._forward_expert_loop(hidden_states, tok, sel_e, slot_weights, counts)
        starts = torch.cumsum(counts, 0) - counts
        pos = torch.arange(num_slots, device=device, dtype=torch.long) - starts.repeat_interleave(counts)

        # (E, tmax, H): rows of one expert are contiguous, padding rows are zero.
        # Padding is harmless -- every GEMM row is independent and padded rows are
        # never read back (we gather with (sel_e, pos), which stays inside counts).
        x_sorted = hidden_states.new_zeros(E, tmax, hidden)
        x_sorted[sel_e, pos] = hidden_states.index_select(0, tok)

        # One batched GEMM for gate and up together, then one for down.
        gate_up = torch.bmm(x_sorted, self._gate_up.transpose(1, 2))
        if self._gate_up_bias is not None:
            gate_up = gate_up + self._gate_up_bias.unsqueeze(1)
        gate, up = gate_up[..., :I], gate_up[..., I:]
        down = torch.bmm(self.act_fn(gate) * up, self._down.transpose(1, 2))

        out_slots = down[sel_e, pos]
        if self._down_bias is not None:
            out_slots = out_slots + self._down_bias[sel_e]
        out_slots = out_slots * slot_weights.unsqueeze(-1).to(out_slots.dtype)

        final = hidden_states.new_zeros(M, hidden)
        final.index_add_(0, tok, out_slots)
        return final

    def extra_repr(self):
        return (
            f"num_experts={self.num_experts}, hidden={self.hidden_dim}, "
            f"intermediate={self.intermediate_dim}, dtype={self._gate_up.dtype}"
        )


class FastMoEBlock(nn.Module):
    """Sparse MoE block whose experts use the vectorized fused layout.

    Mirrors ``Qwen3MoeSparseMoeBlock.forward`` exactly (same gate call and the
    same ``experts(hidden, index, weights)`` contract), so the router is reused
    unchanged when the per-expert block is swapped out.
    """

    def __init__(self, gate, experts):
        super().__init__()
        self.gate = gate
        self.experts = experts

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(flat)
        final = self.experts(flat, selected_experts, routing_weights)
        return final.reshape(batch_size, sequence_length, hidden_dim)


_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


def _expert_list(experts):
    if isinstance(experts, nn.ModuleDict):
        return [experts[key] for key in experts]
    return list(experts)


def _is_per_expert_moe_block(module):
    experts = getattr(module, "experts", None)
    if experts is None or getattr(module, "gate", None) is None:
        return False
    if not isinstance(experts, (nn.ModuleList, nn.ModuleDict)):
        return False
    experts = _expert_list(experts)
    return bool(experts) and all(hasattr(experts[0], name) for name in _PROJ_NAMES)


def _stack_bias(experts, proj_names):
    """Stack per-expert biases, or return None when the experts are bias-free."""
    biases = [
        [getattr(getattr(expert, name), "bias", None) for name in proj_names]
        for expert in experts
    ]
    flat = [bias for row in biases for bias in row]
    if all(bias is None for bias in flat):
        return None
    if any(bias is None for bias in flat):
        raise NotImplementedError(
            "fast MoE requires all experts to agree on whether the projections have a bias"
        )
    return torch.stack([torch.cat(row, dim=0) for row in biases], dim=0)


def stack_experts_from_block(block, dtype=None, sync_free_row_budget=SYNC_FREE_ROW_BUDGET,
                            max_padded_rows=MAX_PADDED_ROWS,
                            assume_distinct_topk=ASSUME_DISTINCT_TOPK):
    """Fuse a per-expert MoE block into ``(E, 2I, H)`` / ``(E, H, I)`` tensors.

    Works for both materialized ``FWTLinear`` experts (quantized evaluation) and
    plain ``nn.Linear`` experts (unquantized layers in partial-quantization
    runs).  Returns ``None`` when ``block`` is not a per-expert MoE block.
    """
    if not _is_per_expert_moe_block(block):
        return None
    experts = _expert_list(block.experts)
    act_fn = getattr(experts[0], "act_fn", None) or F.silu

    gate_probe = _module_weight(experts[0].gate_proj)
    down_probe = _module_weight(experts[0].down_proj)
    inter, hidden = gate_probe.shape
    out_dtype = dtype or gate_probe.dtype

    # Preallocate the fused tensors and fill one expert at a time.  Building a
    # list first and calling torch.stack would keep the per-expert sources *and*
    # the fused copy alive simultaneously (2x peak), which can OOM a nearly full
    # device or a large MoE during the eval-time conversion.
    gate_up = torch.empty(len(experts), 2 * inter, hidden,
                          dtype=out_dtype, device=gate_probe.device)
    down = torch.empty(len(experts), down_probe.shape[0], down_probe.shape[1],
                       dtype=out_dtype, device=down_probe.device)
    for index, expert in enumerate(experts):
        # HF's fused ``gate_up_proj`` keeps gate in the first half, up in the
        # second half (see ``Qwen3MoeExperts.forward``), so match that layout.
        gate_up[index, :inter] = _module_weight(expert.gate_proj)
        gate_up[index, inter:] = _module_weight(expert.up_proj)
        down[index] = _module_weight(expert.down_proj)

    gate_up_bias = _stack_bias(experts, ("gate_proj", "up_proj"))
    down_bias = _stack_bias(experts, ("down_proj",))
    if gate_up_bias is not None:
        gate_up_bias = gate_up_bias.to(out_dtype)
    if down_bias is not None:
        down_bias = down_bias.to(out_dtype)
    return FastMoEExperts(gate_up, down, act_fn, gate_up_bias, down_bias,
                          sync_free_row_budget=sync_free_row_budget,
                          max_padded_rows=max_padded_rows,
                          assume_distinct_topk=assume_distinct_topk)


def _release_expert_storage(experts):
    """Drop the per-expert fp caches now that they live in the fused tensors."""
    if experts is None:
        return
    for expert in _expert_list(experts):
        for sub in expert.modules():
            if isinstance(sub, FWTLinear):
                sub._buffers.pop("_weight_fp", None)
                sub._buffers.pop("packed_weight", None)
                sub._parameters.pop("scale", None)


def convert_moe_block_to_fast(block, dtype=None, sync_free_row_budget=SYNC_FREE_ROW_BUDGET,
                              max_padded_rows=MAX_PADDED_ROWS,
                              assume_distinct_topk=ASSUME_DISTINCT_TOPK):
    """Swap a per-expert MoE block for its vectorized equivalent.

    The old per-expert weights are released right after stacking, so peak memory
    stays at a single copy.  Returns the new block, or ``None`` when ``block`` is
    not a per-expert MoE block.
    """
    if not _is_per_expert_moe_block(block):
        return None
    gate = getattr(block, "gate", None)
    fast_experts = stack_experts_from_block(
        block, dtype=dtype, sync_free_row_budget=sync_free_row_budget,
        max_padded_rows=max_padded_rows, assume_distinct_topk=assume_distinct_topk,
    )
    _release_expert_storage(getattr(block, "experts", None))
    return FastMoEBlock(gate=gate, experts=fast_experts)


def convert_moe_blocks_to_fast(module, dtype=None, verbose=False,
                               sync_free_row_budget=SYNC_FREE_ROW_BUDGET,
                               max_padded_rows=MAX_PADDED_ROWS,
                               assume_distinct_topk=ASSUME_DISTINCT_TOPK):
    """Recursively replace every per-expert MoE block below ``module``.

    Returns the dotted names of the blocks that were converted.
    """
    converted = []
    for name, child in list(module.named_children()):
        if _is_per_expert_moe_block(child):
            setattr(module, name, convert_moe_block_to_fast(
                child, dtype=dtype, sync_free_row_budget=sync_free_row_budget,
                max_padded_rows=max_padded_rows, assume_distinct_topk=assume_distinct_topk,
            ))
            converted.append(name)
            if verbose:
                print(f"[moe_fast_eval] converted {name}")
            continue
        for suffix in convert_moe_blocks_to_fast(
            child, dtype=dtype, verbose=verbose, sync_free_row_budget=sync_free_row_budget,
            max_padded_rows=max_padded_rows, assume_distinct_topk=assume_distinct_topk,
        ):
            converted.append(f"{name}.{suffix}")
    return converted
