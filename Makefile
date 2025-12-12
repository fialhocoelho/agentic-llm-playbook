.PHONY: venv setup kernel test lint format demo_attention demo_generate clean help

# Variables
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
JUPYTER := $(VENV)/bin/jupyter

help:
	@echo "Available targets:"
	@echo "  venv             - Create virtual environment"
	@echo "  setup            - Install dependencies"
	@echo "  kernel           - Install Jupyter kernel"
	@echo "  test             - Run pytest suite"
	@echo "  lint             - Run ruff linter"
	@echo "  format           - Format code with ruff"
	@echo "  demo_attention   - Run attention mechanism demo"
	@echo "  demo_generate    - Run text generation demo"
	@echo "  clean            - Remove venv and cache files"

venv:
	python3 -m venv $(VENV)
	@echo "Virtual environment created. Run 'make setup' to install dependencies."

setup: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

kernel: setup
	$(PYTHON) -m ipykernel install --user --name=llm-playbook --display-name="LLM Playbook"
	@echo "Jupyter kernel 'llm-playbook' installed."

test:
	$(PYTEST) tests/ -v

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) check --fix src/ tests/
	$(RUFF) format src/ tests/

demo_attention:
	$(PYTHON) src/llm_journey/demo/attention_demo.py

demo_generate:
	$(PYTHON) src/llm_journey/demo/generate_demo.py

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Cleaned up virtual environment and cache files."
