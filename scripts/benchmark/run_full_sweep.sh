#!/bin/bash

# Script to run full parameter sweep across all configurations

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Configuration
OUTPUT_DIR=${OUTPUT_DIR:-"results/full_sweep"}
CONFIG_FILE=${CONFIG_FILE:-"config/benchmark/base_config.yaml"}

echo "=========================================="
echo "FL Defense Benchmark - Full Sweep"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Run using Hydra for multirun
# Note: This requires the runner to support Hydra integration
# For now, we'll call Python directly

python -c "
import sys
sys.path.insert(0, 'src')
from experiments import run_experiment_from_config
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

results = run_experiment_from_config(config, '$OUTPUT_DIR')
print('Full sweep completed!')
print(f'Results saved to: $OUTPUT_DIR')
"

echo ""
echo "Full sweep completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Check $OUTPUT_DIR/report.md for summary"
