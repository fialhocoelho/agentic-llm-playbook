"""Tests for attention masking functionality."""

import pytest
import torch
from src.llm_journey.models import ScaledDotProductAttention


def test_attention_with_causal_mask():
    """Test that causal masking prevents attending to future tokens."""
    batch_size = 2
    num_heads = 1
    seq_len = 4
    d_k = 8

    # Create inputs
    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    # Apply attention with mask
    attention = ScaledDotProductAttention(dropout=0.0)
    output, weights = attention(query, key, value, mask)

    # Check that weights are zero for future positions
    weights_np = weights[0, 0].detach().numpy()
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert weights_np[i, j] < 1e-6, f"Future attention at ({i}, {j}) should be masked"


def test_attention_without_mask():
    """Test attention without masking attends to all positions."""
    batch_size = 1
    num_heads = 1
    seq_len = 3
    d_k = 4

    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)

    attention = ScaledDotProductAttention(dropout=0.0)
    output, weights = attention(query, key, value, mask=None)

    # Check that all weights sum to 1 (valid probability distribution)
    for i in range(seq_len):
        assert torch.isclose(
            weights[0, 0, i].sum(), torch.tensor(1.0), atol=1e-5
        ), f"Attention weights at position {i} should sum to 1"


def test_mask_shape_broadcast():
    """Test that mask can be broadcast across batch and head dimensions."""
    batch_size = 2
    num_heads = 4
    seq_len = 5
    d_k = 8

    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Create mask with shape (1, 1, seq_len, seq_len) for broadcasting
    mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))

    attention = ScaledDotProductAttention(dropout=0.0)
    output, weights = attention(query, key, value, mask)

    # Check output shape
    assert output.shape == (batch_size, num_heads, seq_len, d_k)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
