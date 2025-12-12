"""Setup configuration for agentic-llm-playbook."""

from setuptools import setup, find_packages

setup(
    name="llm-journey",
    version="0.1.0",
    description="A 4-week build-first roadmap for modern LLM work",
    author="agentic-llm-playbook contributors",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",
        "matplotlib>=3.7.0",
        "pytest>=7.4.0",
        "ruff>=0.1.0",
        "pre-commit>=3.3.0",
    ],
)
