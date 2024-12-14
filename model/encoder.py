import copy

import torch
from torch import nn

from attention import MultiHeadAttention


class Encoder(nn.Module):

    def __init__(self, n_heads, mask):
        super().__init__()
        self_attention = MultiHeadAttention()
        self.attention_layers = nn.ModuleList([copy.deepcopy(self_attention) for _ in range(n_heads)])

    def forward(self):
        pass
