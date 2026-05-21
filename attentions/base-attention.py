import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model должно делиться на num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):

        B, S, _ = query.shape
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)

        return self.out_proj(attn_output)


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        B, S, _ = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(attn_output)
    

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, dropout=0.0):
        super().__init__()
        
        assert d_model % num_heads == 0
        assert num_heads % num_groups == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // num_groups

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_groups * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_groups * self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        B, S, _ = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        k = k.view(B, S, self.num_groups, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_groups, self.head_dim).transpose(1, 2)
        
        k = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        k = k.reshape(B, self.num_heads, S, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        v = v.reshape(B, self.num_heads, S, self.head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(attn_output)
    

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, latent_dim, dropout=0.0):
        super().__init__()
        
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.latent_dim = latent_dim

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_compress = nn.Linear(d_model, latent_dim)
        self.k_proj = nn.Linear(latent_dim, d_model)
        self.v_proj = nn.Linear(latent_dim, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        
        B, S, _ = query.shape
        q = self.q_proj(query)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        latent = self.kv_compress(key)

        k = self.k_proj(latent)
        v = self.v_proj(latent)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)
    
    
class LinearAttention(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        phi = lambda x: F.elu(x) + 1.0

        Q = phi(query)
        K = phi(key)
        V = value

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1)
            K = K * mask
            V = V * mask

        KV = torch.einsum('b n d, b n e -> b d e', K, V)
        
        K_sum = K.sum(dim=1)

        numerator = torch.einsum('b n d, b d e -> b n e', Q, KV)
  
        denominator = torch.einsum('b n d, b d -> b n', Q, K_sum)
        denominator = denominator.unsqueeze(-1).clamp(min=1e-6)

        output = numerator / denominator
        return self.dropout(output)
    

class SparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size=32, stride=8, dropout=0.0):
        super().__init__()
        
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.stride = stride

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _create_sparse_mask(self, S, device):
        
        mask = torch.zeros(S, S, dtype=torch.bool, device=device)

        for i in range(S):
            start = max(0, i - self.window_size)
            end = min(S, i + self.window_size + 1)
            mask[i, start:end] = True

        global_indices = torch.arange(0, S, self.stride, device=device)
        mask[:, global_indices] = True

        float_mask = torch.zeros(1, 1, S, S, device=device)
        float_mask[:, :, ~mask] = -float('inf')
        return float_mask

    def forward(self, query, key, value, is_causal=False):
        B, S, _ = query.shape
        device = query.device

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        mask = self._create_sparse_mask(S, device)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)
    

class SlidingWindowAttention(nn.Module):
    ...
    

class DeformableAttention(nn.Module):
    ...
    

class AttentionOnAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model должно делиться на num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.info_proj = nn.Linear(d_model, d_model)

        self.gate_proj = nn.Linear(2 * d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        
        B, S, _ = query.shape

        q = self.q_proj(query).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        C = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        I = self.info_proj(query)

        gate_input = torch.cat([C, I], dim=-1)
        G = torch.sigmoid(self.gate_proj(gate_input))

        AoA = G * C + (1 - G) * I

        return self.out_proj(AoA)

    
class CrossAttentionWithMemory(nn.Module):
    ...
