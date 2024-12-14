import numpy as np

import torch
from torch.nn import nn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot Product Attention"""

    def __init__(self, d_model, d_k, d_v, dropout=0.0, mask=False):
        """
        Args:
            d_model (int):
            d_k (int): Dimension of queries and keys. Sequence length.
            d_v (int): _description_
            dropout (float, optional):
            mask (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.query = nn.Linear(d_model, d_k)  # W_q
        self.key = nn.Linear(d_model, d_k)  # W_k
        self.values = nn.Linear(d_model, d_v)  # W_v
        self.dropout = nn.Dropout(dropout)
        self.mask = mask

    def forward(self, q, k, v):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(...)) / torch.sqrt(d_k)

        if self.mask is not None:
            mask = torch.triu(torch.ones(...), diagonal=1)
            mask = mask.unsqueeze(0).repeat(..., 1, 1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=1)
        return torch.matmul()


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads):
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        qkv_weights: torch.Tensor,
        o_weights: torch.Tensor,
        num_heads: int,
        scaled_softmax: float,
    ) -> torch.Tensor:
        """Compute multi-headed attention

        Args:
            hidden_states (torch.Tensor): [batch_size, seq_len, hidden_dim]
            qkv_weights (torch.Tensor): [hidden_dim, 3 x hidden_dim]
            o_weights (torch.Tensor): [hidden_dim, hidden_dim]
            num_heads (int): The number of attention heads
            scaled_softmax (float): The scaling for the softmax

        Returns:
            torch.Tensor: The hidden states with updated multi-headed attention
        """

        batch_size, seq_len, hidden_dim = hidden_states.shape  # hidden_dim = 512
        d = hidden_dim // num_heads  # single-head attention dim = 512 // 8 = 64

        # obtain query, key, value matrices
        qkv = torch.matmul(hidden_states, qkv_weights)  # [B, S, 3D]

        # extract q,k,v across multiple heads
        qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, d)  # [B, S, seq_len, 3, num_heads, d]
        q, k, v = qkv.chunk(3, ...)
        ...

        attn_scores = torch.matmul(q, k.tranpose(-2, -1)) / np.sqrt(d)
        attn_weights = scaled_softmax * torch.matmul(torch.softmax(attn_scores), v)  # [B, S, seq_len, ]

        attn_multi_head = torch.cat(attn_weights, dim=...)

        # mapping back to hidden dim
        o = torch.matmul(attn_multi_head, o_weights)  # [B, S, seq_len, d]
        return o
