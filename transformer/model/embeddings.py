import math

import torch
import torch.nn as nn
from torch import Tensor


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Embedding layer for input tokens

        Args:
            d_model (int): Hidden dimension of the model. The size of the vector
                representations (embeddings / hidden states) used throughout the
                Transformer model.
            vocab_size (int): Size of the vocabulary. Number of unique tokens in the
                input data.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Input to embedding layer: (*)
        # Output from embedding layer: (*, H), where H is the hidden dim of the model.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed input tokens.

        Args:
            x (Tensor): Input tokens of shape `(bs, seq_len)`.

        Returns:
            Tensor: Embedded input of shape `(bs, seq_len, d_model)`.
        """
        # seq_len dimension contains token ids that can be mapped back to unique word
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    pe: Tensor

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        """
        Positional encoding / embeddings for input tokens

        Args:
            d_model (int): Hidden dimension of the model. The size of the vector
                representations (embeddings / hidden states) used throughout the
                Transformer model.
            max_seq_len (int, optional): Maximum sequence length.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # Create positional encodings of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)

        ### REFER TO 3.5 Positional Encoding ###

        # Create tensor of shape (max_seq_len, 1)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)  # Even positions
        pe[:, 1::2] = torch.cos(pos * div_term)  # Odd positions

        # Add batch dimension to positional encodings
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)

        # Tensor is saved in file when model is saved.
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply positional encoding to input embeddings.

        Args:
            x (Tensor): Input embeddings of shape `(bs, seq_len, d_model)`.

        Returns:
            Tensor: Positional encodings of shape `(bs, seq_len, d_model)`.
        """

        # Add positional encodings to input embeddings
        seq_len = x.size(1)
        # Shorten positional encodings if seq_len is greater than max_seq_len
        pe_out = (self.pe[:, :seq_len, :]).requires_grad_(False)
        x = x + pe_out
        return self.dropout(x)
