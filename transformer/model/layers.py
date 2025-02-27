import torch
import torch.nn as nn
from torch import Tensor


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        """
        LayerNorm operates independently on each sample within a batch, unlike
        BatchNorm, which normalizes across the batch dimension. It normalizes the inputs
        across the feature dimension.

        Purpose: Mitigate internal covariate shift thus improving training speed,
        stability, and convergence of the model. Also, improves generalization.

        Args:
            eps (float, optional): Epsilon value to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

        # Two learnable parameters
        self.alpha = nn.Parameter(torch.ones(1))  # Scale parameter (Multiplicative)
        self.bias = nn.Parameter(torch.zeros(1))  # Shift parameter (Additive)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)  # Apply mean to last dimension
        std = x.std(-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        ### In section 3.3 Position-wise Feed-Forward Networks ###

        # 'The dimensionality of input and output is d_model = 512, and the inner-layer
        # has dimensionality d_ff = 2048.'
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        1. (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        2. (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)

        Args:
            x (Tensor): The input tensor. `(batch_size, seq_len, d_model)`
        Returns:
            Tensor: The output tensor. `(batch_size, seq_len, d_model)`
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
        # (bs, seq_len, d_model) -> (bs, seq_len, vocab_size)
        return torch.log_softmax(self.linear(x), dim=-1)
