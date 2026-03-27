#!/bin/bash
# Run all SignGuard experiments for the paper

set -e

echo "======================================"
echo "SignGuard Experiments - All Tables & Figures"
echo "======================================"
echo ""

# Create output directories
mkdir -p figures/tables
mkdir -p figures/plots
mkdir -p figures/data
mkdir -p experiments/results
mkdir -p experiments/cache

echo "Experiment Configuration:"
echo "  Output: experiments/results/"
echo "  Figures: figures/"
echo "  Cache: experiments/cache/"
echo ""

# Parse command line arguments
USE_CACHE=${1:-"true"}
NUM_ROUNDS=${2:-10}

echo "Starting experiments..."
echo ""

# Table 1: Defense Comparison
echo "======================================"
echo "Table 1: Defense Comparison"
echo "======================================"
python3 experiments/table1_defense_comparison.py
echo ""

# Table 2: Attack Success Rate
echo "======================================"
echo "Table 2: Attack Success Rate"
echo "======================================"
python3 experiments/table2_attack_success_rate.py
echo ""

# Table 3: Overhead Analysis
echo "======================================"
echo "Table 3: Overhead Analysis"
echo "======================================"
python3 experiments/table3_overhead_analysis.py
echo ""

# Figure 1: Reputation Evolution
echo "======================================"
echo "Figure 1: Reputation Evolution"
echo "======================================"
python3 experiments/figure1_reputation_evolution.py
echo ""

# Figure 2: Detection ROC
echo "======================================"
echo "Figure 2: Detection ROC Curves"
echo "======================================"
python3 experiments/figure2_detection_roc.py
echo ""

# Figure 3: Privacy-Utility
echo "======================================"
echo "Figure 3: Privacy-Utility Trade-off"
echo "======================================"
python3 experiments/figure3_privacy_utility.py
echo ""

# Ablation Study
echo "======================================"
echo "Ablation Study"
echo "======================================"
python3 experiments/ablation_study.py
echo ""

echo "======================================"
echo "All experiments completed!"
echo "Results saved to: experiments/results/"
echo "Figures saved to: figures/"
echo "======================================"
