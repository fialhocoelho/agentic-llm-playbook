# Micro Training: Overfitting on Tiny Data

This notebook demonstrates training GPTMini on a very small dataset to verify everything works.

## Goals

- Verify model can learn and overfit
- Understand the training loop
- Monitor loss and metrics
- Experiment with hyperparameters

## Topics to Cover

### 1. Dataset Preparation
- Load tiny_corpus.txt
- Create CharDataset
- Build vocabulary
- Create train/val splits (even if tiny)

### 2. Training Loop Basics
- Forward pass
- Loss computation
- Backward pass
- Optimizer step
- Gradient clipping

### 3. Monitoring Training
- Plot training loss
- Track learning rate
- Monitor gradient norms
- Validate periodically

### 4. Overfitting Verification
- Model should achieve very low loss on small dataset
- Generate text samples during training
- Observe memorization behavior

## Exercises

1. Train until near-zero loss on tiny corpus
2. Experiment with learning rates
3. Try different optimizer settings (AdamW, SGD)
4. Implement gradient clipping
5. Add basic learning rate scheduling

## Expected Behavior

With a tiny dataset (e.g., a few sentences):
- Loss should drop quickly
- Model should memorize training data
- Generated text should resemble training data
- Validation loss may not be meaningful (too little data)

## Hyperparameters to Try

```python
config = {
    "batch_size": 4,
    "learning_rate": 3e-4,
    "max_iters": 500,
    "eval_interval": 50,
    "grad_clip": 1.0,
}
```

## Next Steps

Week 2 will cover proper training on larger datasets with:
- Better data loading
- Gradient accumulation
- Mixed precision training
- Distributed training basics
