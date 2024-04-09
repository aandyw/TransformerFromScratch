import torch
from torch.nn import nn


class SelfAttention(nn.Module):
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

    def __init__(self):
        pass
