"""Tests for gradient flow through model components."""

import torch
from llm_journey.models import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    TransformerBlock,
)


def test_attention_gradient_flow():
    """Test that gradients flow through attention mechanism."""
    batch_size = 2
    num_heads = 2
    seq_len = 4
    d_k = 8

    query = torch.randn(batch_size, num_heads, seq_len, d_k, requires_grad=True)
    key = torch.randn(batch_size, num_heads, seq_len, d_k, requires_grad=True)
    value = torch.randn(batch_size, num_heads, seq_len, d_k, requires_grad=True)

    attention = ScaledDotProductAttention(dropout=0.0)
    output, weights = attention(query, key, value)

    # Compute loss and backpropagate
    loss = output.sum()
    loss.backward()

    # Check that gradients exist and are non-zero
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
    assert query.grad.abs().sum() > 0
    assert key.grad.abs().sum() > 0
    assert value.grad.abs().sum() > 0


def test_multi_head_attention_gradient_flow():
    """Test gradient flow through multi-head attention."""
    batch_size = 2
    seq_len = 5
    d_model = 32
    num_heads = 4

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    mha = MultiHeadAttention(d_model, num_heads)
    output = mha(x, x, x)

    loss = output.sum()
    loss.backward()

    # Check input gradients
    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    # Check parameter gradients
    for name, param in mha.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_transformer_block_gradient_flow():
    """Test gradient flow through transformer block."""
    batch_size = 2
    seq_len = 5
    d_model = 32
    num_heads = 4
    d_ff = 64

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    block = TransformerBlock(d_model, num_heads, d_ff)
    output = block(x)

    loss = output.sum()
    loss.backward()

    # Check input gradients
    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    # Check all parameters have gradients
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_gradient_accumulation():
    """Test that gradients accumulate correctly across multiple forward passes."""
    d_model = 16
    num_heads = 2

    mha = MultiHeadAttention(d_model, num_heads)

    # First forward-backward pass
    x1 = torch.randn(1, 3, d_model, requires_grad=True)
    output1 = mha(x1, x1, x1)
    loss1 = output1.sum()
    loss1.backward()

    # Store first gradients
    first_grads = {name: param.grad.clone() for name, param in mha.named_parameters()}

    # Second forward-backward pass (without zero_grad)
    x2 = torch.randn(1, 3, d_model, requires_grad=True)
    output2 = mha(x2, x2, x2)
    loss2 = output2.sum()
    loss2.backward()

    # Check that gradients have accumulated
    for name, param in mha.named_parameters():
        accumulated_grad = param.grad
        first_grad = first_grads[name]
        assert not torch.allclose(accumulated_grad, first_grad), (
            f"Gradient for {name} did not accumulate"
        )
