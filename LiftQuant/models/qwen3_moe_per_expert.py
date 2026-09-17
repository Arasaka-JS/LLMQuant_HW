import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeMLP,
    Qwen3MoeTopKRouter,
)


class PerExpertQwen3MoeSparseMoeBlock(nn.Module):
    """Per-expert MoE block compatible with per-expert checkpoints
    (``mlp.experts.{i}.gate_proj/up_proj/down_proj.weight``).

    Newer transformers versions refactored ``Qwen3MoeSparseMoeBlock`` to use
    fused 3D expert weights (``Qwen3MoeExperts`` with ``gate_up_proj`` /
    ``down_proj`` as ``nn.Parameter``). This class restores the older
    per-expert ``nn.ModuleList[Qwen3MoeMLP]`` layout so that:

      1. per-expert checkpoints load without conversion, and
      2. LiftQuant's per-expert MoE branch (``experts[0]`` / ``experts[1]``)
         keeps working.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.gate = Qwen3MoeTopKRouter(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        _, routing_weights, selected_experts = self.gate(hidden_states)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Only iterate over experts that were actually routed to (top-8 sparse),
        # instead of all num_experts. The per-token order produced by torch.where
        # is row-major and matches the fused Qwen3MoeExperts reference exactly.
        hit_experts = expert_mask.sum(dim=(-1, -2)).nonzero().flatten()

        for expert_idx in hit_experts:
            expert_idx = int(expert_idx)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


def patch_qwen3_moe_per_expert():
    """Replace the fused Qwen3MoE block with the per-expert implementation."""
    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = PerExpertQwen3MoeSparseMoeBlock
