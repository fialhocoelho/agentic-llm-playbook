"""Demo of attention mechanism visualization."""

import torch
import matplotlib.pyplot as plt
from llm_journey.models import ScaledDotProductAttention
from llm_journey.utils import set_seed


def visualize_attention():
    """Visualize attention weights for a simple example."""
    set_seed(42)

    # Create simple query, key, value tensors
    batch_size = 1
    seq_len = 5
    d_k = 8

    query = torch.randn(batch_size, 1, seq_len, d_k)
    key = torch.randn(batch_size, 1, seq_len, d_k)
    value = torch.randn(batch_size, 1, seq_len, d_k)

    # Apply attention
    attention = ScaledDotProductAttention(dropout=0.0)
    output, weights = attention(query, key, value)

    # Visualize attention weights
    plt.figure(figsize=(8, 6))
    plt.imshow(weights[0, 0].detach().numpy(), cmap="viridis")
    plt.colorbar()
    plt.title("Attention Weights")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.tight_layout()
    plt.savefig("attention_weights.png")
    print("Attention weights visualization saved to attention_weights.png")

    # Print shapes
    print(f"\nQuery shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")


if __name__ == "__main__":
    print("Running Attention Mechanism Demo...")
    visualize_attention()
