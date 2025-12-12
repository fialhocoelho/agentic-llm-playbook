"""Demo: Inspect attention weights with and without masking."""

import torch
from llm_journey.models.attention import scaled_dot_product_attention, causal_mask
from llm_journey.utils.seed import seed_everything


def main():
    # Set seed for reproducibility
    seed_everything(42)

    print("=" * 60)
    print("Attention Mechanism Demo")
    print("=" * 60)
    print()

    # Create a small toy sequence
    B, T, d_k = 1, 5, 8  # 1 batch, 5 tokens, 8 dimensions

    # Random embeddings for Q, K, V
    q = torch.randn(B, T, d_k)
    k = torch.randn(B, T, d_k)
    v = torch.randn(B, T, d_k)

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print()

    # Unmasked attention
    print("-" * 60)
    print("UNMASKED ATTENTION")
    print("-" * 60)
    out_unmasked, weights_unmasked = scaled_dot_product_attention(q, k, v)
    print(f"Output shape: {out_unmasked.shape}")
    print(f"Attention weights shape: {weights_unmasked.shape}")
    print()
    print("Attention weights (each row should sum to ~1.0):")
    print(weights_unmasked[0].numpy())
    print()
    print("Row sums:", weights_unmasked[0].sum(dim=-1).numpy())
    print()

    # Masked attention (causal)
    print("-" * 60)
    print("CAUSAL MASKED ATTENTION")
    print("-" * 60)
    mask = causal_mask(T, q.device)
    print("Causal mask (0 = attend, -inf = masked):")
    print(mask.numpy())
    print()

    out_masked, weights_masked = scaled_dot_product_attention(q, k, v, mask)
    print(f"Output shape: {out_masked.shape}")
    print(f"Attention weights shape: {weights_masked.shape}")
    print()
    print("Attention weights (lower triangular, rows sum to ~1.0):")
    print(weights_masked[0].numpy())
    print()
    print("Row sums:", weights_masked[0].sum(dim=-1).numpy())
    print()

    # Verify masking worked
    print("-" * 60)
    print("VERIFICATION")
    print("-" * 60)
    upper_triangle = weights_masked[0].triu(diagonal=1)
    max_upper = upper_triangle.max().item()
    print(f"Max attention weight in upper triangle: {max_upper:.6f}")
    print(
        f"Should be ~0 for proper masking: {'✓ PASS' if max_upper < 1e-6 else '✗ FAIL'}"
    )
    print()


if __name__ == "__main__":
    main()
