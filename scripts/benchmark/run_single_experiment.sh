#!/bin/bash

# Script to run a single experiment with specified configuration

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Default configuration
ATTACK=${ATTACK:-"label_flip"}
DEFENSE=${DEFENSE:-"median"}
ATTACKER_FRAC=${ATTACKER_FRAC:-0.2}
ALPHA=${ALPHA:-1.0}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-"results/single_run"}

echo "=========================================="
echo "FL Defense Benchmark - Single Experiment"
echo "=========================================="
echo "Attack: $ATTACK"
echo "Defense: $DEFENSE"
echo "Attacker Fraction: $ATTACKER_FRAC"
echo "Non-IID Alpha: $ALPHA"
echo "Seed: $SEED"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

# Run experiment using Python module
python -m src.experiments.runner \
    --attack "$ATTACK" \
    --defense "$DEFENSE" \
    --attacker_fraction "$ATTACKER_FRAC" \
    --alpha "$ALPHA" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Experiment completed!"
echo "Results saved to: $OUTPUT_DIR"
