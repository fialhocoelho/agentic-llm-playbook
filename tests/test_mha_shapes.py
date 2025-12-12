"""Test multi-head attention shapes."""

import torch
import pytest
from llm_journey.models.mha import MultiHeadAttention
from llm_journey.utils.seed import seed_everything


def test_mha_output_shape():
    """Verify MHA preserves input shape [B, T, C]."""
    seed_everything(42)

    B, T, d_model = 2, 16, 64
    n_heads = 4

    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(B, T, d_model)

    out = mha(x)

    assert out.shape == (B, T, d_model), (
        f"Expected output shape {(B, T, d_model)}, got {out.shape}"
    )


def test_mha_with_mask():
    """Verify MHA works with attention mask."""
    seed_everything(42)

    from llm_journey.models.attention import causal_mask

    B, T, d_model = 2, 16, 64
    n_heads = 4

    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(B, T, d_model)
    mask = causal_mask(T, x.device)

    out = mha(x, mask)

    assert out.shape == (B, T, d_model), (
        f"Expected output shape {(B, T, d_model)}, got {out.shape}"
    )


def test_mha_different_batch_sizes():
    """Verify MHA works with different batch sizes."""
    seed_everything(42)

    d_model = 64
    n_heads = 4
    T = 16

    mha = MultiHeadAttention(d_model, n_heads)

    for B in [1, 2, 8]:
        x = torch.randn(B, T, d_model)
        out = mha(x)
        assert out.shape == (B, T, d_model)


def test_mha_d_model_divisibility():
    """Verify MHA raises error if d_model not divisible by n_heads."""
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=65, n_heads=4)  # 65 not divisible by 4


def test_mha_forward_backward():
    """Verify MHA supports gradient computation."""
    seed_everything(42)

    B, T, d_model = 2, 8, 64
    n_heads = 4

    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(B, T, d_model, requires_grad=True)

    out = mha(x)
    loss = out.sum()
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None, "Input should have gradients"
    assert mha.q_proj.weight.grad is not None, "MHA weights should have gradients"
