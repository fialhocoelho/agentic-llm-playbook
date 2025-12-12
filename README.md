# agentic-llm-playbook

**agentic-llm-playbook** is a 4-week, build-first roadmap for modern LLM work—covering Transformers from scratch, training fundamentals, post-training (DPO/RLHF/GRPO), and agentic RAG with memory and evaluation. It's in active development 🚀

## Status

**Current Focus:** Week 1 in progress  
**Week 1 Deliverables:** Transformer architecture from scratch with runnable demos

## The 4-Week Journey

### Week 1: Transformers From Scratch
Build the core transformer architecture piece by piece:
- Scaled dot-product attention with causal masking
- Multi-head attention (MHA)
- Pre-LN transformer blocks
- GPT-style mini language model
- Character-level tokenization
- Simple text generation

**Notebooks:**
- `01_attention_playground.ipynb` - Explore attention mechanisms
- `02_gptmini_forward.ipynb` - Forward pass through GPTMini
- `03_micro_train_overfit.ipynb` - Overfit on tiny dataset

### Week 2: Training Systems (Coming Soon)
Training loop fundamentals, optimization, and scaling:
- Training loops and gradient accumulation
- Learning rate schedules and warmup
- Mixed precision training (AMP)
- Checkpointing and resumption
- Basic distributed training concepts

### Week 3: Post-Training Alignment (Coming Soon)
Alignment techniques beyond supervised fine-tuning:
- Direct Preference Optimization (DPO)
- Reinforcement Learning from Human Feedback (RLHF)
- Group Relative Policy Optimization (GRPO)
- Reward modeling basics
- Evaluation of aligned models

### Week 4: Agentic RAG + Evaluation (Coming Soon)
Building agentic systems with retrieval and memory:
- Retrieval-Augmented Generation (RAG)
- Vector stores and embeddings
- Agentic workflows with memory
- Tool use and function calling
- Evaluation harnesses
- Light production concerns (serving, monitoring)

## Quickstart

```bash
# Set up environment
make setup

# Install Jupyter kernel
make kernel

# Run tests
make test

# Try the demos
make demo_attention
make demo_generate
```

## Repository Layout

```
src/llm_journey/         # Main package
  ├── models/            # Attention, MHA, Transformer, GPTMini
  ├── data/              # Dataset utilities
  ├── demo/              # Runnable demo scripts
  └── utils/             # Seeds, device helpers
data/                    # Training corpora
notebooks/               # Jupyter notebooks for exploration
tests/                   # Pytest test suite
notes/                   # Weekly learning notes
```

## Ground Rules

- **Educational first:** Code prioritizes readability and learning over performance
- **Runnable always:** Every week's artifacts should run end-to-end
- **Minimal dependencies:** Use PyTorch, NumPy, and standard tools
- **Test everything:** All components have unit tests
- **Document learnings:** Use `notes/` to capture insights

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow guidelines.

```bash
make lint      # Check code style
make format    # Format code
make test      # Run tests
```

## License

MIT
