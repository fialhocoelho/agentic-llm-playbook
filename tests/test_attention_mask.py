"""Test causal masking in attention."""

import torch
from llm_journey.models.attention import scaled_dot_product_attention, causal_mask
from llm_journey.utils.seed import seed_everything


def test_causal_mask_upper_triangle_is_zero():
    """Verify that masked attention weights in upper triangle are ~0."""
    seed_everything(42)

    B, T, d_k = 2, 8, 16
    q = torch.randn(B, T, d_k)
    k = torch.randn(B, T, d_k)
    v = torch.randn(B, T, d_k)

    # Apply causal mask
    mask = causal_mask(T, q.device)
    _, attn_weights = scaled_dot_product_attention(q, k, v, mask)

    # Check that upper triangle (excluding diagonal) is ~0
    for b in range(B):
        upper_triangle = attn_weights[b].triu(diagonal=1)
        assert upper_triangle.max().item() < 1e-6, (
            f"Upper triangle should be ~0 but got max {upper_triangle.max().item()}"
        )


def test_attention_weights_sum_to_one():
    """Verify that each row of attention weights sums to ~1."""
    seed_everything(42)

    B, T, d_k = 2, 8, 16
    q = torch.randn(B, T, d_k)
    k = torch.randn(B, T, d_k)
    v = torch.randn(B, T, d_k)

    # Test both masked and unmasked
    for mask in [None, causal_mask(T, q.device)]:
        _, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # Each row should sum to 1
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Attention weights rows should sum to 1, got {row_sums}"
        )


def test_causal_mask_shape():
    """Verify causal mask has correct shape."""
    T = 10
    mask = causal_mask(T, torch.device("cpu"))
    assert mask.shape == (T, T), f"Expected shape ({T}, {T}), got {mask.shape}"


def test_causal_mask_structure():
    """Verify causal mask has correct structure (lower triangular)."""
    T = 5
    mask = causal_mask(T, torch.device("cpu"))

    # Lower triangle (including diagonal) should be 0
    lower = torch.tril(torch.ones(T, T))
    assert torch.all(mask[lower == 1] == 0), "Lower triangle should be 0"

    # Upper triangle should be -inf
    upper = torch.triu(torch.ones(T, T), diagonal=1)
    assert torch.all(torch.isinf(mask[upper == 1])), "Upper triangle should be -inf"
    assert torch.all(mask[upper == 1] < 0), "Upper triangle should be negative inf"
