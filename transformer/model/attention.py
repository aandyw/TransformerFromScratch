import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads  # aka. `h`

        assert d_model % num_heads == 0  # Ensure d_model is divisible by num_heads

        self.d_k = d_model // num_heads  # d_k = d_v

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def scaled_dot_product_attention(
        q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, dropout: nn.Dropout | None
    ) -> tuple[Tensor, Tensor]:
        """Compute Scaled Dot Product Attention"""

        d_k = q.shape[-1]

        # (bs, num_heads, seq_len, d_k) -> (bs, num_heads, seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask:
            # For all values in mask == 0 replace with -inf
            scores = scores.masked_fill(mask == 0, float("-inf"))

        scores = torch.softmax(scores, dim=-1)  # (bs, num_heads, seq_len, seq_len)

        if dropout:
            scores = dropout(scores)

        weights = scores @ v  # (bs, num_heads, seq_len, d_k)

        # We return the scores for visualization
        return weights, scores

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        query = self.W_q(q)  # (bs, seq_len, d_model) -> (bs, seq_len, d_model)
        key = self.W_k(k)  # (bs, seq_len, d_model) -> (bs, seq_len, d_model)
        value = self.W_v(v)  # (bs, seq_len, d_model) -> (bs, seq_len, d_model)

        # (bs, seq_len, d_model) -> (bs, seq_len, num_heads, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k)
        # (bs, seq_len, num_heads, d_k) -> (bs, num_heads, seq_len, d_k)
        query = query.transpose(1, 2)

        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        weights, scores = MultiHeadAttention.scaled_dot_product_attention(
            query, key, value, mask, self.dropout
        )

        ### Perform concatenation of the heads ###

        # (bs, num_heads, seq_len, d_k) -> (bs, seq_len, num_heads, d_k)
        weights = weights.transpose(1, 2)

        # (bs, seq_len, num_heads, d_k) -> (bs, seq_len, d_model)
        concat = weights.contiguous().view(
            weights.shape[0], weights.shape[1], self.d_model
        )

        # (bs, seq_len, d_model) -> (bs, seq_len, d_model)
        return self.W_o(concat)
