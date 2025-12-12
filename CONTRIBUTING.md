# Contributing to LLM Journey Playbook

## Weekly Workflow

This is a build-first playbook organized by weeks. When contributing:

1. **Keep commits small and focused** - Each commit should represent one logical change
2. **Run tests before committing** - Use `make test` to verify your changes
3. **Update notes** - Document learnings in the appropriate `notes/weekXX.md`
4. **Keep demos runnable** - All demo scripts should execute end-to-end
5. **Stay minimal** - Code should be educational and readable, not production-grade

## Development Setup

```bash
make setup    # Install dependencies
make kernel   # Set up Jupyter kernel
make test     # Run all tests
```

## Before Submitting

- [ ] Code is formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Tests pass (`make test`)
- [ ] Demos run successfully (`make demo_attention`, `make demo_generate`)
- [ ] Updated relevant notes/weekXX.md with learnings

## Code Style

- Use clear variable names
- Add docstrings to functions and classes
- Keep functions small and focused
- Prioritize readability over cleverness
- Add comments for non-obvious implementation choices

## Testing Philosophy

Tests should:
- Verify shapes and dimensions
- Check gradient flow
- Validate attention masking
- Be fast and deterministic (use fixed seeds)
