# Contributing to agentic-llm-playbook

Thank you for your interest in contributing to the agentic-llm-playbook! This project is a 4-week build-first course covering modern LLM development.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of Python and machine learning concepts

### Setting Up Your Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/agentic-llm-playbook.git
   cd agentic-llm-playbook
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   make venv
   make setup
   ```

3. **Install the Jupyter kernel (optional, for notebook development):**
   ```bash
   make kernel
   ```

4. **Install pre-commit hooks:**
   ```bash
   venv/bin/pre-commit install
   ```

## Development Workflow

### Code Style

We use **ruff** for linting and formatting Python code. Before submitting a pull request:

```bash
make lint    # Check for linting issues
make format  # Auto-format code
```

### Testing

All code changes should include appropriate tests. Run the test suite with:

```bash
make test
```

Write tests in the `tests/` directory following the existing structure:
- `test_masks.py` - Tests for attention masking
- `test_shapes.py` - Tests for tensor shape validation
- `test_gradients.py` - Tests for gradient computation

### Project Structure

```
agentic-llm-playbook/
├── src/llm_journey/        # Main source code
│   ├── utils/              # Utility functions
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model implementations
│   └── demo/               # Demo scripts
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks for tutorials
├── tests/                  # Test suite
└── requirements.txt        # Python dependencies
```

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)

### Suggesting Enhancements

We welcome suggestions! Open an issue describing:
- The enhancement you'd like to see
- Why it would be useful
- Any implementation ideas

### Submitting Pull Requests

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Ensure quality:**
   ```bash
   make lint
   make format
   make test
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Guidelines

- Keep changes focused and atomic
- Write clear commit messages
- Update documentation for user-facing changes
- Ensure all tests pass
- Follow existing code style and conventions

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Check existing issues and pull requests
- Review the documentation in `notebooks/`

Thank you for contributing to making LLM education more accessible! 🚀
