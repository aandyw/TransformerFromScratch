import torch
import torch.nn as nn
from torch import Tensor


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        """
        LayerNorm operates independently on each sample within a batch, unlike
        BatchNorm, which normalizes across the batch dimension. It normalizes the
        inputs across the feature dimension.

        Purpose: Mitigate internal covariate shift thus improving training speed,
        stability, and convergence of the model. Also, improves generalization.

        Args:
            eps (float, optional): Epsilon value to avoid division by zero.
                Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

        # Two learnable parameters
        self.alpha = nn.Parameter(torch.ones(1))  # Scale parameter (Multiplicative)
        self.bias = nn.Parameter(torch.zeros(1))  # Shift parameter (Additive)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer norm to last dimension of the input tensor.

        Args:
            x (Tensor): `(bs, seq_len, d_model)`.

        Returns:
            Tensor: `(bs, seq_len, d_model)`.
        """
        # Apply mean & std to last dimension
        mean = x.mean(-1, keepdim=True)  # (bs, seq_len, 1)
        std = x.std(-1, keepdim=True)  # (bs, seq_len, 1)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        ### REFER TO Section 3.3 Position-wise Feed-Forward Networks ###

        # 'The dimensionality of input and output is d_model = 512, and the inner-layer
        # has dimensionality d_ff = 2048.'
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies linear transformations with ReLU activation function between.
            1. (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
            2. (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)

        Args:
            x (Tensor): `(bs, seq_len, d_model)`.

        Returns:
            Tensor: `(bs, seq_len, d_model)`.
        """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        """
        Residual connection with layer normalization.

        Args:
            x (Tensor): `(bs, seq_len, d_model)`.
            sublayer (nn.Module): The intermediate layer to wrap w/ residual connection.

        Returns:
            Tensor: `(bs, seq_len, d_model)`.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class LinearLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Linear Layer is a projection layer that converts the embedding into the
        vocabulary.

        Args:
            d_model (int): The size of the model's hidden dimension.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply projection on embeddings.
        Output will be a log probability distribution over the vocabulary.

        Args:
            x (Tensor): `(bs, seq_len, d_model)`.

        Returns:
            Tensor: `(bs, seq_len, vocab_size)`.
        """

        # (bs, seq_len, d_model) -> (bs, seq_len, vocab_size)
        # return log probabilities not probabilities
        return torch.log_softmax(self.linear(x), dim=-1)
