"""Attention mechanisms for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, dropout: float = 0.1):
        """
        Initialize the attention module.

        Args:
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None
    ):
        """
        Apply scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch, heads, seq_len, d_k)
            key: Key tensor of shape (batch, heads, seq_len, d_k)
            value: Value tensor of shape (batch, heads, seq_len, d_v)
            mask: Optional mask tensor

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        d_k = query.size(-1)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        """
        Apply multi-head attention.

        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor of shape (batch, seq_len, d_model)
            value: Value tensor of shape (batch, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape to (batch, heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        x, _ = self.attention(Q, K, V, mask)

        # Reshape and apply output projection
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(x)

        return output
