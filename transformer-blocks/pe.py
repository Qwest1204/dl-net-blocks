import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.size(1)
        return x + self.pe[:, :S, :]
    

class LearnableAbsolutePositionalEmbeddings(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.size(1)
        positions = torch.arange(S, device=x.device).unsqueeze(0)
        pos_emb = self.embeddings(positions)
        return x + pos_emb


class RotaryPositionEmbedding(nn.Module):
    """
    rope = RotaryPositionEmbedding(head_dim=64)
    q, k = ...
    cos, sin = rope.get_cos_sin(q.size(2), q.device)
    q_rot = RotaryPositionEmbedding.apply_rotary(q, cos, sin)
    k_rot = RotaryPositionEmbedding.apply_rotary(k, cos, sin)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    def get_cos_sin(self, seq_len: int, device: torch.device):
        cos = self.cos_cached[:, :, :seq_len, :].to(device)
        sin = self.sin_cached[:, :, :seq_len, :].to(device)
        return cos, sin
    

class LearnedRelativePositionalEmbedding(nn.Module):
    ...