"""
Plotting functions for benchmark visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Use non-interactive backend for server environments
matplotlib.use("Agg")


def set_plot_style() -> None:
    """Set consistent plotting style for all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def plot_heatmap(
    data: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    metric_name: str,
    title: str,
    save_path: str,
    cmap: str = "RdYlGn_r",
    fmt: str = ".3f",
    annot: bool = True,
) -> None:
    """
    Plot heatmap of results.

    Args:
        data: 2D array of values to plot
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        metric_name: Name of metric (for colorbar label)
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap name
        fmt: Format string for annotations
        annot: Whether to annotate cells with values
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)

    # Add annotations
    if annot:
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(
                    j,
                    i,
                    fmt % data[i, j],
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    # Labels and title
    ax.set_xlabel("Defense Method")
    ax.set_ylabel("Attack / Attacker Fraction")
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_convergence(
    results: List[Dict[str, Any]],
    metric: str = "test_accuracy",
    save_path: Optional[str] = None,
    defense_names: Optional[List[str]] = None,
) -> None:
    """
    Plot convergence curves for different defenses.

    Args:
        results: List of result dictionaries with 'rounds' and metric keys
        metric: Metric to plot (e.g., 'test_accuracy', 'asr')
        save_path: Path to save figure
        defense_names: Optional list of defense names for legend
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, result in enumerate(results):
        rounds = result.get("rounds", [])
        values = result.get(metric, [])

        if defense_names and i < len(defense_names):
            label = defense_names[i]
        else:
            label = f"Method {i + 1}"

        ax.plot(rounds, values, marker="o", linewidth=2, markersize=4, label=label)

    ax.set_xlabel("Round")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} vs Training Round")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_bar_comparison(
    data: Dict[str, List[float]],
    metric_name: str,
    title: str,
    save_path: str,
    ylabel: Optional[str] = None,
    rotate_labels: bool = True,
) -> None:
    """
    Plot bar chart comparing different methods.

    Args:
        data: Dictionary mapping method names to lists of values
        metric_name: Name of metric being plotted
        title: Plot title
        save_path: Path to save figure
        ylabel: Y-axis label (defaults to metric_name)
        rotate_labels: Whether to rotate x-axis labels
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(data.keys())
    means = [np.mean(values) for values in data.values()]
    stds = [np.std(values, ddof=1) for values in data.values()]

    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8)

    # Color bars differently
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45 if rotate_labels else 0, ha="right")
    ax.set_ylabel(ylabel or metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    results: List[Dict[str, Any]],
    save_path: str,
    method_names: Optional[List[str]] = None,
) -> None:
    """
    Plot ROC curves for different methods.

    Args:
        results: List of dictionaries with 'fpr' and 'tpr' keys
        save_path: Path to save figure
        method_names: Optional list of method names for legend
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, result in enumerate(results):
        fpr = result.get("fpr", [])
        tpr = result.get("tpr", [])
        auc = result.get("auc", 0.0)

        if method_names and i < len(method_names):
            label = f"{method_names[i]} (AUC = {auc:.3f})"
        else:
            label = f"Method {i + 1} (AUC = {auc:.3f})"

        ax.plot(fpr, tpr, linewidth=2, label=label)

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curves(
    results: List[Dict[str, Any]],
    save_path: str,
    method_names: Optional[List[str]] = None,
) -> None:
    """
    Plot Precision-Recall curves for different methods.

    Args:
        results: List of dictionaries with 'precision' and 'recall' keys
        save_path: Path to save figure
        method_names: Optional list of method names for legend
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, result in enumerate(results):
        precision = result.get("precision", [])
        recall = result.get("recall", [])
        auprc = result.get("auprc", 0.0)

        if method_names and i < len(method_names):
            label = f"{method_names[i]} (AUPRC = {auprc:.3f})"
        else:
            label = f"Method {i + 1} (AUPRC = {auprc:.3f})"

        ax.plot(recall, precision, linewidth=2, label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_attacker_fraction_vs_metric(
    attacker_fractions: List[float],
    metrics: Dict[str, List[float]],
    metric_name: str,
    save_path: str,
) -> None:
    """
    Plot metric as a function of attacker fraction.

    Args:
        attacker_fractions: List of attacker fractions
        metrics: Dictionary mapping defense names to metric values
        metric_name: Name of metric being plotted
        save_path: Path to save figure
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for defense_name, values in metrics.items():
        ax.plot(attacker_fractions, values, marker="o", linewidth=2, label=defense_name)

    ax.set_xlabel("Attacker Fraction")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{metric_name.replace('_', ' ').title()} vs Attacker Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_non_iid_impact(
    alpha_values: List[float],
    metrics: Dict[str, List[float]],
    metric_name: str,
    save_path: str,
    log_scale: bool = True,
) -> None:
    """
    Plot impact of non-IID level (alpha) on metric.

    Args:
        alpha_values: List of Dirichlet alpha values
        metrics: Dictionary mapping defense names to metric values
        metric_name: Name of metric being plotted
        save_path: Path to save figure
        log_scale: Whether to use log scale for x-axis
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for defense_name, values in metrics.items():
        ax.plot(alpha_values, values, marker="o", linewidth=2, label=defense_name)

    ax.set_xlabel("Dirichlet α (Non-IID Level)")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{metric_name.replace('_', ' ').title()} vs Non-IID Level")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_figure(
    results: Dict[str, Any],
    save_path: str,
) -> None:
    """
    Create a multi-panel summary figure.

    Args:
        results: Dictionary with all results
        save_path: Path to save figure
    """
    set_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Accuracy vs Attacker Fraction
    ax1 = axes[0, 0]
    if "attacker_fractions" in results and "accuracy_vs_fraction" in results:
        for defense, values in results["accuracy_vs_fraction"].items():
            ax1.plot(results["attacker_fractions"], values, marker="o", label=defense)
        ax1.set_xlabel("Attacker Fraction")
        ax1.set_ylabel("Clean Accuracy")
        ax1.set_title("Accuracy vs Attacker Fraction")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Panel 2: ASR vs Attacker Fraction
    ax2 = axes[0, 1]
    if "attacker_fractions" in results and "asr_vs_fraction" in results:
        for defense, values in results["asr_vs_fraction"].items():
            ax2.plot(results["attacker_fractions"], values, marker="o", label=defense)
        ax2.set_xlabel("Attacker Fraction")
        ax2.set_ylabel("Attack Success Rate")
        ax2.set_title("ASR vs Attacker Fraction")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Panel 3: Accuracy vs Non-IID Level
    ax3 = axes[1, 0]
    if "alpha_values" in results and "accuracy_vs_alpha" in results:
        for defense, values in results["accuracy_vs_alpha"].items():
            ax3.plot(results["alpha_values"], values, marker="o", label=defense)
        ax3.set_xlabel("Dirichlet α")
        ax3.set_ylabel("Clean Accuracy")
        ax3.set_xscale("log")
        ax3.set_title("Accuracy vs Non-IID Level")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Panel 4: Bar chart comparison
    ax4 = axes[1, 1]
    if "final_metrics" in results:
        defenses = list(results["final_metrics"].keys())
        accuracies = [results["final_metrics"][d]["accuracy"] for d in defenses]
        asrs = [results["final_metrics"][d]["asr"] for d in defenses]

        x = np.arange(len(defenses))
        width = 0.35

        ax4.bar(x - width/2, accuracies, width, label="Accuracy", alpha=0.8)
        ax4.bar(x + width/2, asrs, width, label="ASR", alpha=0.8)

        ax4.set_xlabel("Defense Method")
        ax4.set_ylabel("Value")
        ax4.set_title("Final Metrics Comparison")
        ax4.set_xticks(x)
        ax4.set_xticklabels(defenses, rotation=45, ha="right")
        ax4.legend()
        ax4.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
