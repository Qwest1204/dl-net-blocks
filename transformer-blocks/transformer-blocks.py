import torch
import torch.nn as nn

class SimpleFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias   = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias
        return x_norm
    

class MixtureOfExperts(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2,
                 expert: nn.Module = SimpleFFN, expert_args: dict = None,
                 router_bias: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        expert_args = expert_args or {}
        self.experts = nn.ModuleList([
            expert(d_model, **expert_args) for _ in range(num_experts)
        ])

        self.router = nn.Linear(d_model, num_experts, bias=router_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, d = x.shape
        
        router_logits = self.router(x)

        topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1

        router_weights = F.softmax(topk_logits, dim=-1)

        output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            if expert_mask.sum() == 0:
                continue

            tokens_for_expert = x[expert_mask]
            expert_out = self.experts[expert_idx](tokens_for_expert)
            weights_for_expert = router_weights[expert_mask]
            chosen_slot_mask = (topk_indices[expert_mask] == expert_idx)
            w = (weights_for_expert * chosen_slot_mask).sum(dim=-1, keepdim=True)

            output[expert_mask] += w * expert_out

        if self.training:
            
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for k in range(self.top_k):
                expert_counts.scatter_add_(0, topk_indices[:, :, k].reshape(-1),
                                           torch.ones_like(router_weights[:, :, k].reshape(-1)))
            expert_counts = expert_counts / (B * S)
            router_probs = F.softmax(router_logits, dim=-1).mean(dim=(0, 1))
            router_loss = (expert_counts * router_probs).sum() * self.num_experts
        else:
            router_loss = torch.tensor(0.0, device=x.device)

        return output, router_loss
    

