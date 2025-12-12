# Week 1: Transformers From Scratch

## Overview

Week 1 focuses on building the core transformer architecture from first principles. The goal is understanding, not performance.

## Key Concepts

### Attention Mechanism
- **Scaled Dot-Product Attention**: Q·K^T / sqrt(d_k), then softmax, then multiply by V
- **Why Scaling?**: Prevents dot products from getting too large, which would push softmax into regions with tiny gradients
- **Causal Masking**: Prevents attending to future tokens (autoregressive property)

### Multi-Head Attention (MHA)
- Split d_model into n_heads parallel attention operations
- Each head learns different relationships (e.g., syntactic vs semantic)
- Concatenate and project back to d_model
- Allows model to attend to different positions and subspaces simultaneously

### Transformer Block Architecture
- **Pre-LN (Pre-Layer Normalization)**: Normalize before attention/MLP, not after
  - More stable training than Post-LN
  - Easier to stack more layers
- **Residual Connections**: x + f(x) allows gradients to flow directly through
- **Feed-Forward Network**: Two linear layers with GELU activation
  - Typically d_ff = 4 * d_model
  - Processes each position independently

### GPT-Style Language Model
- Token embeddings + learned positional embeddings
- Stack of transformer blocks
- Final layer norm + language modeling head
- **Weight Tying**: Share weights between token embedding and LM head
  - Reduces parameters
  - Often improves performance

## Implementation Details

### Embedding Strategy
- Token embedding: vocabulary → d_model
- Positional embedding: position → d_model (learned, not sinusoidal for Week 1)
- Simply add them together (broadcasting handles batch dimension)

### Causal Masking
- Upper triangular mask set to -inf
- After adding mask, softmax pushes masked positions to ~0
- Essential for autoregressive generation

### Generation
- Autoregressive: generate one token at a time
- Temperature: controls randomness (higher = more random)
- Top-k: only sample from k most likely tokens
- Greedy: always pick argmax (deterministic but boring)

## Testing Philosophy

Week 1 tests focus on correctness, not performance:

1. **Shape Tests**: Verify tensor dimensions throughout
2. **Mask Tests**: Ensure causal masking works correctly
3. **Gradient Tests**: Verify backprop flows through all components
4. **Numerical Tests**: Check attention weights sum to 1, etc.

## Common Pitfalls & Solutions

### Issue: Attention weights don't sum to 1
- **Cause**: Forgot to apply softmax, or applied twice
- **Fix**: Single softmax on scores after masking

### Issue: Upper triangle not masked
- **Cause**: Mask has wrong sign or wrong shape
- **Fix**: Use -inf for masked positions, 0 for unmasked

### Issue: Gradients are None
- **Cause**: Tensors not marked as requires_grad, or detached somewhere
- **Fix**: Check requires_grad, avoid in-place operations

### Issue: Shape mismatch in MHA
- **Cause**: Incorrect view/reshape/transpose operations
- **Fix**: Carefully track shapes: (B,T,C) → (B,T,H,D) → (B,H,T,D)

## Key Learnings

1. **Attention is all you need**: No RNNs, no convolutions, just attention and MLPs
2. **Parallelization**: Unlike RNNs, all positions can be processed in parallel
3. **Inductive Bias**: Minimal architectural inductive bias, model learns from data
4. **Positional Info**: Must explicitly add position info (no inherent ordering)

## Next Steps (Week 2)

- Implement proper training loop
- Add gradient accumulation
- Learning rate scheduling (warmup + decay)
- Mixed precision training (AMP)
- Checkpointing and resumption
- Train on larger datasets (e.g., TinyStories, OpenWebText)

## Resources & References

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (GPT-2 paper)
- Andrej Karpathy's minGPT and nanoGPT
- The Illustrated Transformer (Jay Alammar)
- Transformer Circuits Thread (Anthropic)

## Experiments to Try

1. Visualize attention patterns on simple sequences
2. Compare Pre-LN vs Post-LN training stability
3. Experiment with different numbers of heads
4. Try different positional embedding strategies (learned vs sinusoidal)
5. Implement layer-wise learning rate decay
6. Add dropout and see effect on overfitting
