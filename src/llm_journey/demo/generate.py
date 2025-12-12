"""Demo: Generate text with GPTMini (random weights for Week 1)."""

import torch
from llm_journey.models.gptmini import GPTMini
from llm_journey.utils.seed import seed_everything
from llm_journey.utils.device import get_device


def main():
    # Set seed for reproducibility
    seed_everything(42)
    device = get_device(prefer_cuda=False)  # Use CPU for demo

    print("=" * 60)
    print("GPTMini Text Generation Demo")
    print("=" * 60)
    print()

    # Create a tiny model
    vocab_size = 50  # Small vocab for demo
    model = GPTMini(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        max_seq_len=128,
        dropout=0.0,  # No dropout for demo
    )
    model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print()

    # Create a simple prompt (just token indices)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)  # (1, 3)
    print(f"Prompt tokens: {prompt[0].tolist()}")
    print()

    # Generate with different settings
    print("-" * 60)
    print("Generation with temperature=1.0, no top-k")
    print("-" * 60)
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"Generated tokens: {generated[0].tolist()}")
    print()

    print("-" * 60)
    print("Generation with temperature=0.8, top_k=10")
    print("-" * 60)
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=10)
    print(f"Generated tokens: {generated[0].tolist()}")
    print()

    print("-" * 60)
    print("Forward pass test with dummy targets")
    print("-" * 60)
    # Test forward pass with loss computation
    dummy_input = torch.randint(0, vocab_size, (2, 16), device=device)  # (2, 16)
    dummy_targets = torch.randint(0, vocab_size, (2, 16), device=device)

    logits, loss = model(dummy_input, dummy_targets)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print()

    print("=" * 60)
    print("Demo complete! ✓")
    print("=" * 60)
    print()
    print("Note: Model has random weights (Week 1). For meaningful text,")
    print("train the model on real data in Week 2!")
    print()


if __name__ == "__main__":
    main()
