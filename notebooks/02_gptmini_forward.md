# GPTMini Forward Pass

This notebook walks through a complete forward pass of the GPTMini model.

## Topics to Cover

### 1. Model Architecture Overview
- Token embeddings
- Positional embeddings
- Transformer blocks
- Language modeling head

### 2. Step-by-Step Forward Pass
- Input tokenization
- Embedding lookup
- Adding positional information
- Passing through transformer blocks
- Computing output logits

### 3. Understanding Model Components
- Pre-layer normalization (Pre-LN)
- Residual connections
- Feed-forward networks (MLP)
- Layer normalization

### 4. Output Interpretation
- Logits to probabilities
- Temperature scaling
- Top-k and top-p sampling

## Exercises

1. Trace a single token through the entire model
2. Compare Pre-LN vs Post-LN architectures
3. Analyze the effect of different layer counts
4. Experiment with different embedding dimensions

## Visualization Ideas

- Plot embedding space (PCA/t-SNE)
- Visualize residual stream at each layer
- Show logit distributions before softmax
- Compare attention patterns across layers

## Next Steps

Move to `03_micro_train_overfit.ipynb` to train the model on a tiny dataset.
