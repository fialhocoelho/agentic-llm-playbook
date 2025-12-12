"""Transformer block implementation."""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Layer normalization and residual connections
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ):
        """
        Initialize the transformer block.

        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            dropout: Dropout rate
        """
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, x, x, mask)
        x = self.ln1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)

        return x
