"""Demo of text generation with a simple model."""

import torch
from src.llm_journey.data import SimpleTokenizer
from src.llm_journey.utils import set_seed


def generate_text_demo():
    """Demonstrate text generation flow (without trained model)."""
    set_seed(42)

    # Sample text
    corpus = "Hello, world! This is a demo of text generation."

    # Create tokenizer
    tokenizer = SimpleTokenizer(corpus)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {sorted(tokenizer.char_to_idx.keys())}")

    # Encode and decode
    text = "Hello"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"\nOriginal text: '{text}'")
    print(f"Encoded tokens: {tokens}")
    print(f"Decoded text: '{decoded}'")

    # Demonstrate generation concept
    print("\n--- Text Generation Concept ---")
    print("In a trained model, we would:")
    print("1. Encode input text to tokens")
    print("2. Pass through transformer layers")
    print("3. Sample from output distribution")
    print("4. Decode sampled token to text")
    print("5. Repeat for desired length")

    # Show tensor shapes for generation
    batch_size = 1
    seq_len = 10
    d_model = 64

    print(f"\nExample tensor shapes during generation:")
    print(f"  Input tokens: ({batch_size}, {seq_len})")
    print(f"  Embedded input: ({batch_size}, {seq_len}, {d_model})")
    print(f"  After transformer: ({batch_size}, {seq_len}, {d_model})")
    print(f"  Logits: ({batch_size}, {seq_len}, {tokenizer.vocab_size})")


if __name__ == "__main__":
    print("Running Text Generation Demo...")
    generate_text_demo()
