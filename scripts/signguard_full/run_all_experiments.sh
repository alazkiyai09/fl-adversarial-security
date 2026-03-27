#!/bin/bash
# Run all SignGuard experiments for the paper

set -e

echo "======================================"
echo "SignGuard Experiments"
echo "======================================"

# Create results directory
mkdir -p results/raw results/processed figures/tables figures/plots

# Set default values
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"results/experiments"}
NUM_ROUNDS=${NUM_ROUNDS:-100}
NUM_GPUS=${NUM_GPUS:-1}

echo "Experiment configuration:"
echo "  Output directory: $EXPERIMENT_DIR"
echo "  Number of rounds: $NUM_ROUNDS"
echo "  Number of GPUs: $NUM_GPUS"
echo ""

# Table 1: Defense comparison
echo "======================================"
echo "Running Table 1: Defense Comparison"
echo "======================================"
python experiments/table1_defense_comparison.py

# Table 2: Attack success rate
echo "======================================"
echo "Running Table 2: Attack Success Rate"
echo "======================================"
python experiments/table2_attack_success_rate.py

# Table 3: Overhead analysis
echo "======================================"
echo "Running Table 3: Overhead Analysis"
echo "======================================"
python experiments/table3_overhead_analysis.py

# Figure 1: Reputation evolution
echo "======================================"
echo "Running Figure 1: Reputation Evolution"
echo "======================================"
python experiments/figure1_reputation_evolution.py

# Figure 2: Detection ROC curves
echo "======================================"
echo "Running Figure 2: Detection ROC"
echo "======================================"
python experiments/figure2_detection_roc.py

# Figure 3: Privacy-utility trade-off
echo "======================================"
echo "Running Figure 3: Privacy-Utility"
echo "======================================"
python experiments/figure3_privacy_utility.py

# Ablation study
echo "======================================"
echo "Running Ablation Study"
echo "======================================"
python experiments/ablation_study.py

echo "======================================"
echo "All experiments completed!"
echo "Results saved to: $EXPERIMENT_DIR"
echo "======================================"
