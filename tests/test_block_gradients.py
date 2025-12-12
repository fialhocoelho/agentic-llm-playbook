"""Test gradient flow through transformer blocks."""

import torch
import pytest
from llm_journey.models.transformer import TransformerBlock
from llm_journey.models.gptmini import GPTMini
from llm_journey.utils.seed import seed_everything


def test_transformer_block_gradients():
    """Verify gradients flow through transformer block."""
    seed_everything(42)
    
    B, T, d_model = 2, 8, 64
    n_heads = 4
    d_ff = 256
    
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(B, T, d_model, requires_grad=True)
    
    out = block(x)
    loss = out.sum()
    loss.backward()
    
    # Check that input has gradients
    assert x.grad is not None, "Input should have gradients"
    
    # Check that at least some parameters have gradients
    has_grad = False
    for name, param in block.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "At least some parameters should have non-zero gradients"


def test_gptmini_forward_backward():
    """Verify full forward and backward pass through GPTMini."""
    seed_everything(42)
    
    vocab_size = 50
    B, T = 2, 16
    
    model = GPTMini(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
    )
    
    # Create random inputs and targets
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))
    
    # Forward pass
    logits, loss = model(idx, targets)
    
    # Backward pass
    loss.backward()
    
    # Check shapes
    assert logits.shape == (B, T, vocab_size)
    assert loss.ndim == 0  # Scalar loss
    
    # Check that at least some parameters have gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "At least some model parameters should have non-zero gradients"


def test_gradient_accumulation_across_blocks():
    """Verify gradients accumulate properly through multiple blocks."""
    seed_everything(42)
    
    vocab_size = 50
    model = GPTMini(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=3,
        n_heads=2,
        d_ff=64,
    )
    
    idx = torch.randint(0, vocab_size, (1, 8))
    targets = torch.randint(0, vocab_size, (1, 8))
    
    _, loss = model(idx, targets)
    loss.backward()
    
    # Check that embedding layer has gradients (it's at the start)
    assert model.token_embedding.weight.grad is not None
    
    # Check that final layer has gradients
    assert model.ln_f.weight.grad is not None
    
    # Check that at least one middle block has gradients
    middle_block = model.blocks[1]
    assert middle_block.ln1.weight.grad is not None


def test_no_gradients_in_eval_mode():
    """Verify that gradient computation can be disabled."""
    seed_everything(42)
    
    vocab_size = 50
    model = GPTMini(vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=2)
    model.eval()
    
    idx = torch.randint(0, vocab_size, (1, 8))
    
    with torch.no_grad():
        logits, _ = model(idx)
    
    # Should not raise error when computing without gradients
    assert logits.requires_grad is False
