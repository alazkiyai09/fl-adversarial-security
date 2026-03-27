#!/bin/bash
# Setup script for SignGuard development environment

set -e

echo "======================================"
echo "SignGuard Environment Setup"
echo "======================================"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda env create -f environment.yml || conda env update -f environment.yml
    echo "Activating signguard environment..."
    eval "$(conda shell.bash hook)"
    conda activate signguard
else
    echo "Conda not found. Setting up with pip..."
    pip install -r requirements.txt
fi

# Install package in development mode
echo "Installing SignGuard in development mode..."
pip install -e .

# Install development tools
echo "Installing development tools..."
pip install black isort mypy flake8 pytest-cov pre-commit

# Setup pre-commit hooks (if git repo)
if [ -d .git ]; then
    echo "Setting up pre-commit hooks..."
    pre-commit install
fi

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/credit_card
mkdir -p data/synthetic_bank
mkdir -p checkpoints
mkdir -p results/raw
mkdir -p results/processed
mkdir -p figures/tables
mkdir -p figures/plots

echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To activate the environment:"
if command -v conda &> /dev/null; then
    echo "  conda activate signguard"
else
    echo "  source venv/bin/activate  (if using virtualenv)"
fi
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "To run linting:"
echo "  black signguard/ tests/"
echo "  isort signguard/ tests/"
echo "  mypy signguard/"
echo "======================================"
