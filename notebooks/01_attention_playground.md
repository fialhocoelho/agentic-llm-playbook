# Attention Playground

This notebook explores the attention mechanism in depth.

## Topics to Cover

### 1. Scaled Dot-Product Attention
- Implement and visualize attention scores
- Understand the role of scaling by sqrt(d_k)
- Visualize attention weights as heatmaps

### 2. Attention Masking
- Compare unmasked vs causal masked attention
- Visualize how masking prevents "looking ahead"
- Explore different mask patterns

### 3. Multi-Head Attention
- Implement attention with multiple heads
- Visualize what different heads attend to
- Understand how heads capture different relationships

### 4. Attention Patterns in Practice
- Load pre-computed attention weights
- Analyze attention patterns in real sentences
- Identify common patterns (e.g., attending to previous token, subject-verb agreement)

## Exercises

1. Modify attention to use different scoring functions
2. Implement relative position attention
3. Experiment with different mask patterns
4. Visualize attention heads as heatmaps for sample text

## Next Steps

Move to `02_gptmini_forward.ipynb` to see how attention fits into a complete transformer model.
