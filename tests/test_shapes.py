"""Tests for tensor shape validation in models."""

import pytest
import torch
from llm_journey.models import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    TransformerBlock,
)


def test_scaled_attention_output_shape():
    """Test that scaled attention produces correct output shapes."""
    batch_size = 2
    num_heads = 4
    seq_len = 10
    d_k = 16

    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)

    attention = ScaledDotProductAttention()
    output, weights = attention(query, key, value)

    assert output.shape == (batch_size, num_heads, seq_len, d_k)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_multi_head_attention_output_shape():
    """Test that multi-head attention produces correct output shape."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8

    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model, num_heads)
    output = mha(x, x, x)

    assert output.shape == (batch_size, seq_len, d_model)


def test_transformer_block_output_shape():
    """Test that transformer block preserves input shape."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8
    d_ff = 256

    x = torch.randn(batch_size, seq_len, d_model)

    block = TransformerBlock(d_model, num_heads, d_ff)
    output = block(x)

    assert output.shape == (batch_size, seq_len, d_model)


def test_multi_head_attention_divisibility():
    """Test that d_model must be divisible by num_heads."""
    d_model = 64
    num_heads = 7  # Not a divisor of 64

    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model, num_heads)


def test_various_sequence_lengths():
    """Test that models work with different sequence lengths."""
    d_model = 32
    num_heads = 4

    mha = MultiHeadAttention(d_model, num_heads)

    for seq_len in [1, 5, 10, 20]:
        x = torch.randn(1, seq_len, d_model)
        output = mha(x, x, x)
        assert output.shape == (1, seq_len, d_model)
