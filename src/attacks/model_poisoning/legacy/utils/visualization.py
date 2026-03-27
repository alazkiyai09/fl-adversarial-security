"""
Visualization tools for attack analysis.

Generates plots for accuracy curves, detectability, and attack comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import pandas as pd


def plot_convergence(
    metrics_history: Dict[str, List[Dict]],
    output_path: str = None
):
    """
    Plot convergence curves for different attack strategies.

    Args:
        metrics_history: Dictionary mapping attack name to list of metrics per round
        output_path: Path to save plot (if None, display)
    """
    plt.figure(figsize=(12, 5))

    # Accuracy over time
    plt.subplot(1, 2, 1)
    for attack_name, history in metrics_history.items():
        rounds = range(len(history))
        accuracies = [m.get("accuracy", 0.0) for m in history]
        plt.plot(rounds, accuracies, label=attack_name, linewidth=2)

    plt.xlabel("Federated Learning Round")
    plt.ylabel("Test Accuracy")
    plt.title("Model Accuracy Over Rounds")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss over time
    plt.subplot(1, 2, 2)
    for attack_name, history in metrics_history.items():
        rounds = range(len(history))
        losses = [m.get("loss", 0.0) for m in history]
        plt.plot(rounds, losses, label=attack_name, linewidth=2)

    plt.xlabel("Federated Learning Round")
    plt.ylabel("Test Loss")
    plt.title("Model Loss Over Rounds")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved convergence plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_detectability(
    detection_results: Dict[str, Dict],
    output_path: str = None
):
    """
    Plot detectability metrics for different attacks.

    Args:
        detection_results: Dictionary mapping attack name to detectability metrics
        output_path: Path to save plot
    """
    attack_names = list(detection_results.keys())
    detection_rates = [detection_results[name].get("detection_rate", 0.0) for name in attack_names]
    false_positive_rates = [detection_results[name].get("false_positive_rate", 0.0) for name in attack_names]

    x = np.arange(len(attack_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, detection_rates, width, label="Detection Rate", color="coral")
    bars2 = ax.bar(x + width/2, false_positive_rates, width, label="False Positive Rate", color="lightblue")

    ax.set_xlabel("Attack Strategy")
    ax.set_ylabel("Rate")
    ax.set_title("Attack Detectability Analysis")
    ax.set_xticks(x)
    ax.set_xticklabels(attack_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved detectability plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_attack_comparison(
    comparison_df: pd.DataFrame,
    output_path: str = None
):
    """
    Plot comprehensive comparison of attack strategies.

    Args:
        comparison_df: DataFrame with attack comparison metrics
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Final Accuracy Comparison
    ax1 = axes[0, 0]
    ax1.bar(comparison_df["attack"], comparison_df["final_accuracy"], color="steelblue")
    ax1.set_ylabel("Final Accuracy")
    ax1.set_title("Final Model Accuracy by Attack")
    ax1.set_xticklabels(comparison_df["attack"], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Convergence Speed
    ax2 = axes[0, 1]
    colors = ["green" if x != float("inf") else "red" for x in comparison_df["convergence_round"]]
    ax2.bar(comparison_df["attack"], comparison_df["convergence_round"], color=colors)
    ax2.set_ylabel("Rounds to Converge")
    ax2.set_title("Convergence Speed by Attack (Red = Never Converged)")
    ax2.set_xticklabels(comparison_df["attack"], rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Detection Rate
    ax3 = axes[1, 0]
    ax3.bar(comparison_df["attack"], comparison_df["detection_rate"], color="coral")
    ax3.set_ylabel("Detection Rate")
    ax3.set_title("Attack Detection Rate")
    ax3.set_xticklabels(comparison_df["attack"], rotation=45, ha="right")
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Computational Overhead
    ax4 = axes[1, 1]
    if "train_time_avg" in comparison_df.columns:
        ax4.bar(comparison_df["attack"], comparison_df["train_time_avg"], color="purple")
        ax4.set_ylabel("Avg Training Time (s)")
        ax4.set_title("Computational Overhead by Attack")
        ax4.set_xticklabels(comparison_df["attack"], rotation=45, ha="right")
        ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_l2_norm_distribution(
    update_history: List[Dict],
    output_path: str = None
):
    """
    Plot distribution of L2 norms per round to visualize attack patterns.

    Args:
        update_history: History of client updates with L2 norms
        output_path: Path to save plot
    """
    rounds = []
    honest_norms = []
    malicious_norms = []

    for round_data in update_history:
        round_num = round_data.get("round", 0)
        updates = round_data.get("updates", [])

        for update in updates:
            rounds.append(round_num)
            if update.get("is_malicious", False):
                malicious_norms.append(update.get("l2_norm", 0.0))
            else:
                honest_norms.append(update.get("l2_norm", 0.0))

    plt.figure(figsize=(12, 5))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(
        [r for r, _ in zip(rounds, honest_norms)],
        honest_norms,
        alpha=0.5,
        label="Honest Clients",
        color="blue"
    )
    plt.scatter(
        [r for r, _ in zip(rounds, malicious_norms)],
        malicious_norms,
        alpha=0.7,
        label="Malicious Clients",
        color="red",
        marker="x"
    )
    plt.xlabel("Round")
    plt.ylabel("L2 Norm")
    plt.title("L2 Norm Distribution by Client Type")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Box plot comparison
    plt.subplot(1, 2, 2)
    data_to_plot = [
        honest_norms if honest_norms else [0],
        malicious_norms if malicious_norms else [0]
    ]
    plt.boxplot(data_to_plot, labels=["Honest", "Malicious"])
    plt.ylabel("L2 Norm")
    plt.title("L2 Norm Distribution Comparison")
    plt.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved L2 norm plot to {output_path}")
    else:
        plt.show()

    plt.close()
