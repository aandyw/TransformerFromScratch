import copy

import torch
from torch import nn

from layers.attention import MultiHeadAttention


class Encoder(nn.Module):

    def __init__(self, n_heads, mask):
        super().__init__()
        self_attention = MultiHeadAttention()
        self.attention_layers = nn.ModuleList([copy.deepcopy(self_attention) for _ in range()])

    def forward(self):
        pass
