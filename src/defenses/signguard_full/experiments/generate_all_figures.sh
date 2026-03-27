#!/bin/bash
# Regenerate all figures and tables from cached results

set -e

echo "======================================"
echo "Regenerating SignGuard Figures"
echo "======================================"
echo ""

# Check if results exist
if [ ! -d "experiments/results" ]; then
    echo "Error: experiments/results/ directory not found!"
    echo "Please run scripts/run_all_experiments.sh first."
    exit 1
fi

# Figure 1: Reputation evolution
echo "Regenerating Figure 1: Reputation Evolution"
python3 experiments/figure1_reputation_evolution.py

# Figure 2: Detection ROC
echo "Regenerating Figure 2: Detection ROC"
python3 experiments/figure2_detection_roc.py

# Figure 3: Privacy-utility
echo "Regenerating Figure 3: Privacy-Utility"
python3 experiments/figure3_privacy_utility.py

# Ablation
echo "Regenerating Ablation Study"
python3 experiments/ablation_study.py

echo ""
echo "======================================"
echo "All figures regenerated!"
echo "Saved to: figures/"
echo "======================================"
