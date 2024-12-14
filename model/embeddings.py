import torch
import torch.nn as nn
from torch import Tensor


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """Embedding layer for input tokens

        Args:
            d_model (int): Hidden dimension of the model. The size of the vector
                representations (embeddings / hidden states) used throughout the Transformer model.
            vocab_size (int): Size of the vocabulary. Number of unique tokens in the input data.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Input to embedding layer: (*)
        # Output from embedding layer: (*, H), where H is the hidden dim of the model.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tokens of shape `(batch_size, seq_len)`.

        Returns:
            Tensor: Embedded input of shape `(batch_size, seq_len, d_model)`.
        """
        return self.embedding(x) * self.d_model**0.5


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int = 512, dropout: float = 0.1):
        """Positional encoding / embeddings for input tokens

        Args:
            d_model (int): Hidden dimension of the model. The size of the vector
                representations (embeddings / hidden states) used throughout the
                Transformer model.
            seq_len (int, optional): Length of input sequences. Defaults to 512.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create positional encodings of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        ### REFER TO 3.5 Positional Encoding in [paper](https://arxiv.org/pdf/1706.03762) ###
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(Tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)  # Even positions
        pe[:, 1::2] = torch.cos(pos * div_term)  # Odd positions

        # Add batch dimension to positional encodings
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Tensor is saved in file when model is saved.
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input embeddings of shape `(batch_size, seq_len, d_model)`.

        Returns:
            Tensor: Positional encodings of shape `(batch_size, seq_len, d_model)`.
        """

        # Add positional encodings to input embeddings
        x = x + self.pe[...]
        return self.dropout(x)