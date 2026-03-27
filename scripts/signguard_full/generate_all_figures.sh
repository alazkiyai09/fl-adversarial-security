#!/bin/bash
# Regenerate all figures and tables from cached results

set -e

echo "======================================"
echo "Regenerating SignGuard Figures"
echo "======================================"

# Check if results exist
if [ ! -d "results/processed" ]; then
    echo "Error: results/processed directory not found!"
    echo "Please run scripts/run_all_experiments.sh first."
    exit 1
fi

# Figure 1: Reputation evolution
echo "Generating Figure 1: Reputation Evolution"
python experiments/figure1_reputation_evolution.py --use-cached

# Figure 2: Detection ROC
echo "Generating Figure 2: Detection ROC"
python experiments/figure2_detection_roc.py --use-cached

# Figure 3: Privacy-utility
echo "Generating Figure 3: Privacy-Utility"
python experiments/figure3_privacy_utility.py --use-cached

# Tables
echo "Generating Tables"
python experiments/table1_defense_comparison.py --use-cached
python experiments/table2_attack_success_rate.py --use-cached
python experiments/table3_overhead_analysis.py --use-cached

# Ablation
echo "Generating Ablation Study"
python experiments/ablation_study.py --use-cached

echo "======================================"
echo "All figures regenerated!"
echo "Saved to: figures/"
echo "======================================"
