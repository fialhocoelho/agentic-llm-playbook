"""Multi-head attention module."""

import torch
import torch.nn as nn
from llm_journey.models.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.

    Splits the input into multiple heads, applies attention in parallel,
    then concatenates and projects the results.
    """

    def __init__(self, d_model: int, n_heads: int):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply multi-head attention.

        Args:
            x: Input tensor of shape (B, T, d_model)
            attn_mask: Optional mask of shape (T, T)

        Returns:
            Output tensor of shape (B, T, d_model)
        """
        B, T, C = x.shape

        # Project and split into heads
        # (B, T, d_model) -> (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply attention
        # (B, n_heads, T, d_head)
        out, _ = scaled_dot_product_attention(q, k, v, attn_mask)

        # Merge heads
        # (B, n_heads, T, d_head) -> (B, T, n_heads, d_head) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        out = self.out_proj(out)

        return out
