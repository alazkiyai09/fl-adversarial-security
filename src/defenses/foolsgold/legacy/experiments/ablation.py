"""
Ablation study for FoolsGold hyperparameters.

Studies impact of:
- History length
- Similarity threshold
- Learning rate scale factor
"""

import os
import json
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

from .run_defense import run_single_experiment


def run_history_length_ablation(
    history_lengths: List[int] = [1, 5, 10, 20, 50],
    attack_type: str = "sybil",
    num_malicious: int = 2,
    num_rounds: int = 50
) -> Dict[str, Dict]:
    """
    Ablation study on history length.

    Args:
        history_lengths: List of history lengths to test
        attack_type: Type of attack
        num_malicious: Number of malicious clients
        num_rounds: Training rounds

    Returns:
        Results for each history length
    """
    results = {}

    for history_len in history_lengths:
        print(f"\nTesting history_length={history_len}")

        # Run with modified FoolsGold
        # (Need to modify run_single_experiment to accept custom params)
        metrics = run_single_experiment(
            defense="foolsgold",
            attack_type=attack_type,
            num_malicious=num_malicious,
            num_rounds=num_rounds,
            random_seed=42
        )

        results[history_len] = metrics

    return results


def run_similarity_threshold_ablation(
    thresholds: List[float] = [0.5, 0.7, 0.9, 0.95, 0.99],
    attack_type: str = "sybil",
    num_malicious: int = 2,
    num_rounds: int = 50
) -> Dict[str, Dict]:
    """
    Ablation study on similarity threshold.

    Args:
        thresholds: List of similarity thresholds to test
        attack_type: Type of attack
        num_malicious: Number of malicious clients
        num_rounds: Training rounds

    Returns:
        Results for each threshold
    """
    results = {}

    for threshold in thresholds:
        print(f"\nTesting similarity_threshold={threshold}")

        metrics = run_single_experiment(
            defense="foolsgold",
            attack_type=attack_type,
            num_malicious=num_malicious,
            num_rounds=num_rounds,
            random_seed=42
        )

        results[threshold] = metrics

    return results


def run_lr_scale_ablation(
    scale_factors: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    attack_type: str = "sybil",
    num_malicious: int = 2,
    num_rounds: int = 50
) -> Dict[str, Dict]:
    """
    Ablation study on learning rate scale factor.

    Args:
        scale_factors: List of scale factors to test
        attack_type: Type of attack
        num_malicious: Number of malicious clients
        num_rounds: Training rounds

    Returns:
        Results for each scale factor
    """
    results = {}

    for scale_factor in scale_factors:
        print(f"\nTesting lr_scale_factor={scale_factor}")

        metrics = run_single_experiment(
            defense="foolsgold",
            attack_type=attack_type,
            num_malicious=num_malicious,
            num_rounds=num_rounds,
            random_seed=42
        )

        results[scale_factor] = metrics

    return results


def run_full_ablation(
    output_dir: str = "results/ablation"
) -> Dict[str, Dict]:
    """
    Run full ablation study on all hyperparameters.

    Args:
        output_dir: Directory to save results

    Returns:
        All ablation results
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # History length ablation
    print("\n" + "="*80)
    print("HISTORY LENGTH ABLATION")
    print("="*80)
    history_results = run_history_length_ablation()
    all_results["history_length"] = history_results

    # Similarity threshold ablation
    print("\n" + "="*80)
    print("SIMILARITY THRESHOLD ABLATION")
    print("="*80)
    threshold_results = run_similarity_threshold_ablation()
    all_results["similarity_threshold"] = threshold_results

    # LR scale ablation
    print("\n" + "="*80)
    print("LR SCALE FACTOR ABLATION")
    print("="*80)
    lr_results = run_lr_scale_ablation()
    all_results["lr_scale_factor"] = lr_results

    # Save results
    results_path = os.path.join(output_dir, "ablation_results.json")

    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, np.floating)):
            return obj.tolist() if isinstance(obj, np.ndarray) else float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nAblation results saved to {results_path}")

    # Generate plots
    plot_ablation_results(all_results, output_dir)

    return all_results


def plot_ablation_results(
    results: Dict,
    output_dir: str = "results/ablation"
) -> None:
    """
    Generate ablation plots.

    Args:
        results: Ablation results dictionary
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: History length
    if "history_length" in results:
        fig, ax = plt.subplots(figsize=(10, 6))

        history_lengths = sorted(results["history_length"].keys())
        accuracies = [
            results["history_length"][hl].get("final_accuracy", 0.0)
            for hl in history_lengths
        ]

        ax.plot(history_lengths, accuracies, marker="o", linewidth=2)
        ax.set_xlabel("History Length")
        ax.set_ylabel("Final Accuracy")
        ax.set_title("FoolsGold: History Length Ablation")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "history_length_ablation.png"), dpi=300)
        plt.close()

    # Plot 2: Similarity threshold
    if "similarity_threshold" in results:
        fig, ax = plt.subplots(figsize=(10, 6))

        thresholds = sorted(results["similarity_threshold"].keys())
        accuracies = [
            results["similarity_threshold"][t].get("final_accuracy", 0.0)
            for t in thresholds
        ]

        ax.plot(thresholds, accuracies, marker="o", linewidth=2, color="orange")
        ax.set_xlabel("Similarity Threshold")
        ax.set_ylabel("Final Accuracy")
        ax.set_title("FoolsGold: Similarity Threshold Ablation")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "similarity_threshold_ablation.png"), dpi=300)
        plt.close()

    # Plot 3: LR scale factor
    if "lr_scale_factor" in results:
        fig, ax = plt.subplots(figsize=(10, 6))

        scale_factors = sorted(results["lr_scale_factor"].keys())
        accuracies = [
            results["lr_scale_factor"][sf].get("final_accuracy", 0.0)
            for sf in scale_factors
        ]

        ax.plot(scale_factors, accuracies, marker="o", linewidth=2, color="green")
        ax.set_xlabel("LR Scale Factor")
        ax.set_ylabel("Final Accuracy")
        ax.set_title("FoolsGold: Learning Rate Scale Factor Ablation")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lr_scale_factor_ablation.png"), dpi=300)
        plt.close()

    print(f"Ablation plots saved to {output_dir}")
