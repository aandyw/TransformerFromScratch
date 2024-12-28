from attention import MultiHeadAttention
from layers import FeedForwardBlock, LayerNormalization, ResidualConnection
from torch import Tensor, nn


### ENCODER ###
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
        """
        Args:
            x (Tensor): The input to the encoder.
            src_mask (Tensor): The mask for the source language.

        Returns:
            Tensor: A tensor of `(batch_size, seq_len, d_model)` represents a sequence
                of context-rich embeddings that encode the input sequence's semantic and
                positional information.
        """
        for layer in self.layers:
            x = layer(x, src_mask)

        # Apply a final layer normalization after all encoder blocks
        return self.norm(x)


### DECODER ###
class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttention,
        cross_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        dropout: float = 0.1,
    ):
        """
        Decoder block contains:
            1. (Masked Multi-Head Attention) A self-attention block where `qkv` come
                from decoder's input embedding.
            2. (Multi-Head Attention) A cross-attention block where `q` come from
                decoder and `k`,`v` come from encoder outputs.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """
        Decoder block ussed for machine-translation to go from source to target language.

        Args:
            x (Tensor): The input to the decoder block.
            encoder_output (Tensor): The output from the encoder.
            src_mask (Tensor): The mask used for the source language (e.g. English).
            tgt_mask (Tensor): The mask used for the target language (e.g. German).
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x,
                encoder_output,
                encoder_output,
                src_mask,
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            x (Tensor): The input to the decoder block.
            encoder_output (Tensor): The output from the encoder.
            src_mask (Tensor): The mask used for the source language (e.g. English).
            tgt_mask (Tensor): The mask used for the target language (e.g. German).

        Returns:
            Tensor: _description_
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


### TRANSFORMER ###
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
