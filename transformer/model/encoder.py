import torch.nn as nn
from torch import Tensor


from attention import MultiHeadAttention
from layers import FeedForwardBlock, ResidualConnection, LayerNormalization


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)

        # Apply a final layer normalization after all encoder blocks
        return self.norm(x)
