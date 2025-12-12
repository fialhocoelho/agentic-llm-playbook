.PHONY: help venv setup kernel test lint format clean demo_attention demo_generate

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
JUPYTER := $(VENV)/bin/jupyter

help:
	@echo "Available targets:"
	@echo "  make venv           - Create virtual environment"
	@echo "  make setup          - Install dependencies"
	@echo "  make kernel         - Install Jupyter kernel"
	@echo "  make test           - Run pytest"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Run ruff formatter"
	@echo "  make demo_attention - Run attention inspection demo"
	@echo "  make demo_generate  - Run text generation demo"
	@echo "  make clean          - Remove virtual environment and caches"

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: venv
	$(PIP) install -r requirements.txt
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

kernel: setup
	$(PYTHON) -m ipykernel install --user --name=llm-journey --display-name="LLM Journey"

test:
	PYTHONPATH=src $(PYTEST) tests/ -v

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) format src/ tests/

demo_attention:
	PYTHONPATH=src $(PYTHON) src/llm_journey/demo/inspect_attention.py

demo_generate:
	PYTHONPATH=src $(PYTHON) src/llm_journey/demo/generate.py

clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
