import torch.nn as nn
from attention import MultiHeadAttention
from embeddings import InputEmbeddings, PositionalEncoding
from layers import FeedForwardBlock, LayerNormalization, LinearLayer, ResidualConnection
from torch import Tensor


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
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: LinearLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self, tgt: Tensor, encoder_output: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: Tensor) -> Tensor:
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,  # hidden dimension of the model
    num_blocks: int = 6,  # number of encoder and decoder blocks
    num_heads: int = 8,  # number of attention heads
    d_ff: int = 2048,  # size of the feed-forward layer
    dropout: float = 0.1,
) -> Transformer:
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_layers = nn.ModuleList(
        [
            EncoderBlock(
                MultiHeadAttention(d_model, num_heads, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout,
            )
            for _ in range(num_blocks)
        ]
    )

    # Create decoder blocks
    decoder_layers = nn.ModuleList(
        [
            DecoderBlock(
                MultiHeadAttention(d_model, num_heads, dropout),
                MultiHeadAttention(d_model, num_heads, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout,
            )
            for _ in range(num_blocks)
        ]
    )

    # Create encoder and decoder
    encoder = Encoder(encoder_layers)
    decoder = Decoder(decoder_layers)

    # Create projection layer
    projection_layer = LinearLayer(d_model, tgt_vocab_size)

    # Create transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
