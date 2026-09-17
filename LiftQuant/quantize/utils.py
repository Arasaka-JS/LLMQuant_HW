
import torch
import torch.nn as nn
from scipy import linalg
import functools

def get_parameters(model, use_shift=True):
    params = []
    for n, m in model.named_parameters():
        if n.find('alpha') > -1:
            params.append(m)
    return iter(params) 



class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     
      





def get_act_means(model, dataloader, num_samples, bsz, keys, attention_mask,position_embeddings):
    model.eval()
    device = next(model.parameters()).device
    act_disturb = {}

    #拼接所有样本的激活值，保留通道维度
    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach().cpu()
        if name in act_disturb:
            act_disturb[name] = torch.cat((act_disturb[name], tensor.to(torch.float32).to('cpu')), dim=0)
        else:
            act_disturb[name] = tensor.to(torch.float32).to('cpu')

    def stat_input_hook(m, x, y, name):
        # 捕获输入
        
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        for key in keys:
            if isinstance(m, nn.Linear) and ((key in name)):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=key))
                )

    for i in range(num_samples//bsz):
        model(dataloader[i*bsz:(i+1)*bsz].to(device), attention_mask=attention_mask,position_embeddings=position_embeddings)


    for h in hooks:
        h.remove()

    return act_disturb


def get_moe_act_means(model, dataloader, num_samples, bsz, keys, attention_mask, position_embeddings):
    """Collect attention activation stats + per-expert routing statistics for MoE.

    Returns ``(act_disturb, moe_stats)`` where ``act_disturb`` holds the same
    attention-keyed activation tensors as :func:`get_act_means`, and
    ``moe_stats`` is a dict with ``all_in`` (flattened MoE input hidden states),
    ``all_sel`` (router selected expert indices per token), ``num_experts`` and
    ``counts`` (routing count per expert).
    """
    model.eval()
    device = next(model.parameters()).device
    act_disturb = {}
    mlp_inputs = []
    mlp_selected = []

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach().cpu()
        if name in act_disturb:
            act_disturb[name] = torch.cat((act_disturb[name], tensor.to(torch.float32).to('cpu')), dim=0)
        else:
            act_disturb[name] = tensor.to(torch.float32).to('cpu')

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    def mlp_input_hook(m, x, y):
        inp = x[0] if isinstance(x, tuple) else x
        mlp_inputs.append(inp.view(-1, inp.shape[-1]).detach().cpu())

    def gate_output_hook(m, x, y):
        # Qwen3MoeTopKRouter.forward returns (router_logits, router_scores, router_indices)
        selected = y[-1] if isinstance(y, tuple) else y
        mlp_selected.append(selected.detach().cpu())

    hooks = []
    for name, m in model.named_modules():
        for key in keys:
            if isinstance(m, nn.Linear) and (key in name):
                hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=key)))

    mlp = getattr(model, 'mlp', None)
    if mlp is not None:
        hooks.append(mlp.register_forward_hook(mlp_input_hook))
        if hasattr(mlp, 'gate'):
            hooks.append(mlp.gate.register_forward_hook(gate_output_hook))

    for i in range(num_samples // bsz):
        model(dataloader[i * bsz:(i + 1) * bsz].to(device), attention_mask=attention_mask, position_embeddings=position_embeddings)

    for h in hooks:
        h.remove()

    num_experts = getattr(mlp, 'num_experts', None)
    if mlp is None or num_experts is None or len(mlp_inputs) == 0:
        return act_disturb, None

    all_in = torch.cat(mlp_inputs, dim=0)
    all_sel = torch.cat(mlp_selected, dim=0)
    counts = torch.zeros(num_experts, dtype=torch.long)
    for e in range(num_experts):
        counts[e] = (all_sel == e).any(dim=1).sum().item()

    moe_stats = {
        'all_in': all_in,
        'all_sel': all_sel,
        'num_experts': num_experts,
        'counts': counts,
    }
    return act_disturb, moe_stats
