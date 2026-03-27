"""
Visualization functions for attack impact analysis.

This module provides functions to create plots and visualizations
that help analyze the impact of label flipping attacks.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')


def plot_accuracy_over_rounds(
    baseline_history: Dict[str, List],
    attacked_history: Dict[str, List],
    attack_name: str = "Label Flipping Attack",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot accuracy over training rounds for baseline vs attacked model.

    Args:
        baseline_history: Training history without attack
        attacked_history: Training history under attack
        attack_name: Name of the attack for the title
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_rounds = baseline_history.get("round", [])
    baseline_accs = baseline_history.get("global_accuracy", [])

    attacked_rounds = attacked_history.get("round", [])
    attacked_accs = attacked_history.get("global_accuracy", [])

    # Plot baseline
    ax.plot(
        baseline_rounds,
        baseline_accs,
        'o-',
        linewidth=2,
        markersize=4,
        label='Baseline (No Attack)',
        color='green'
    )

    # Plot attacked
    ax.plot(
        attacked_rounds,
        attacked_accs,
        's-',
        linewidth=2,
        markersize=4,
        label=f'Under {attack_name}',
        color='red'
    )

    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Global Model Accuracy: Baseline vs Under Attack', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add final accuracy annotations
    if baseline_accs:
        final_baseline = baseline_accs[-1]
        ax.text(
            0.02, 0.98,
            f'Final Baseline Acc: {final_baseline:.4f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9
        )

    if attacked_accs:
        final_attacked = attacked_accs[-1]
        degradation = baseline_accs[-1] - final_attacked if baseline_accs else 0
        ax.text(
            0.02, 0.90,
            f'Final Attacked Acc: {final_attacked:.4f}\nDegradation: {degradation:.4f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=9
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_per_class_accuracy(
    baseline_history: Dict[str, List],
    attacked_history: Dict[str, List],
    attack_name: str = "Label Flipping Attack",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot per-class accuracy (fraud vs legitimate) over rounds.

    Args:
        baseline_history: Training history without attack
        attacked_history: Training history under attack
        attack_name: Name of the attack for the title
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    rounds = baseline_history.get("round", [])

    # Baseline per-class accuracy
    baseline_fraud = baseline_history.get("accuracy_fraud", [])
    baseline_legit = baseline_history.get("accuracy_legitimate", [])

    # Attacked per-class accuracy
    attacked_fraud = attacked_history.get("accuracy_fraud", [])
    attacked_legit = attacked_history.get("accuracy_legitimate", [])

    # Plot baseline
    if baseline_fraud:
        ax.plot(
            rounds[:len(baseline_fraud)],
            baseline_fraud,
            'o-',
            linewidth=2,
            markersize=4,
            label='Fraud (Baseline)',
            color='darkgreen'
        )

    if baseline_legit:
        ax.plot(
            rounds[:len(baseline_legit)],
            baseline_legit,
            '^-',
            linewidth=2,
            markersize=4,
            label='Legitimate (Baseline)',
            color='lightgreen'
        )

    # Plot attacked
    if attacked_fraud:
        ax.plot(
            rounds[:len(attacked_fraud)],
            attacked_fraud,
            's-',
            linewidth=2,
            markersize=4,
            label=f'Fraud ({attack_name})',
            color='darkred'
        )

    if attacked_legit:
        ax.plot(
            rounds[:len(attacked_legit)],
            attacked_legit,
            'v-',
            linewidth=2,
            markersize=4,
            label=f'Legitimate ({attack_name})',
            color='lightcoral'
        )

    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy: Baseline vs Under Attack', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_attacker_fraction_impact(
    results: Dict[float, Dict[str, List]],
    baseline_accuracy: float,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot final accuracy vs attacker fraction.

    Args:
        results: Dictionary mapping attacker fraction to history
        baseline_accuracy: Accuracy without any attack
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    fractions = sorted(results.keys())
    final_accuracies = []
    degradations = []

    for frac in fractions:
        history = results[frac]
        accs = history.get("global_accuracy", [])
        final_acc = accs[-1] if accs else 0.0
        final_accuracies.append(final_acc)
        degradations.append(baseline_accuracy - final_acc)

    # Plot accuracy
    ax.plot(
        fractions,
        final_accuracies,
        'o-',
        linewidth=2,
        markersize=8,
        color='red',
        label='Final Accuracy'
    )

    # Add baseline line
    ax.axhline(
        y=baseline_accuracy,
        color='green',
        linestyle='--',
        linewidth=2,
        label=f'Baseline Accuracy ({baseline_accuracy:.4f})'
    )

    ax.set_xlabel('Attacker Fraction', fontsize=12)
    ax.set_ylabel('Final Accuracy', fontsize=12)
    ax.set_title('Impact of Attacker Fraction on Model Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add degradation annotations
    for i, (frac, acc, deg) in enumerate(zip(fractions, final_accuracies, degradations)):
        ax.annotate(
            f'{deg:.3f}',
            (frac, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            color='darkred'
        )

    # Annotate degradation as percentage
    ax.text(
        0.98, 0.02,
        'Numbers indicate\ndegradation from baseline',
        transform=ax.transAxes,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_attack_type_comparison(
    results: Dict[str, Dict[str, List]],
    baseline_accuracy: float,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Compare different attack types side by side.

    Args:
        results: Dictionary mapping attack names to histories
        baseline_accuracy: Accuracy without any attack
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for attack_name, history in results.items():
        rounds = history.get("round", [])
        accs = history.get("global_accuracy", [])

        ax.plot(
            rounds,
            accs,
            'o-',
            linewidth=2,
            markersize=4,
            label=attack_name
        )

    # Add baseline line
    if rounds:
        ax.axhline(
            y=baseline_accuracy,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Baseline ({baseline_accuracy:.4f})'
        )

    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Comparison of Different Attack Types', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_comparison(
    baseline_history: Dict[str, List],
    attacked_history: Dict[str, List],
    threshold: float = 0.01,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize convergence behavior of baseline vs attacked training.

    Args:
        baseline_history: Training history without attack
        attacked_history: Training history under attack
        threshold: Convergence threshold
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the plot
    """
    from src.attacks.data_poisoning.legacy.metrics.attack_metrics import calculate_convergence_delay

    convergence_metrics = calculate_convergence_delay(
        baseline_history,
        attacked_history,
        threshold=threshold
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_rounds = baseline_history.get("round", [])
    baseline_accs = baseline_history.get("global_accuracy", [])
    attacked_rounds = attacked_history.get("round", [])
    attacked_accs = attacked_history.get("global_accuracy", [])

    # Plot accuracies
    ax.plot(
        baseline_rounds,
        baseline_accs,
        'o-',
        linewidth=2,
        markersize=4,
        label='Baseline',
        color='green'
    )

    ax.plot(
        attacked_rounds,
        attacked_accs,
        's-',
        linewidth=2,
        markersize=4,
        label='Attacked',
        color='red'
    )

    # Mark convergence points
    if convergence_metrics["baseline_convergence_round"]:
        baseline_conv_round = convergence_metrics["baseline_convergence_round"]
        baseline_conv_idx = baseline_conv_round - 1
        if baseline_conv_idx < len(baseline_accs):
            ax.axvline(
                x=baseline_conv_round,
                color='green',
                linestyle=':',
                linewidth=2,
                alpha=0.7,
                label=f'Baseline Converges (Round {baseline_conv_round})'
            )

    if convergence_metrics["attacked_convergence_round"]:
        attacked_conv_round = convergence_metrics["attacked_convergence_round"]
        attacked_conv_idx = attacked_conv_round - 1
        if attacked_conv_idx < len(attacked_accs):
            ax.axvline(
                x=attacked_conv_round,
                color='red',
                linestyle=':',
                linewidth=2,
                alpha=0.7,
                label=f'Attacked Converges (Round {attacked_conv_round})'
            )

    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Convergence Comparison: Baseline vs Attacked', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add convergence delay text
    if convergence_metrics["convergence_delay"] is not None:
        delay = convergence_metrics["convergence_delay"]
        ax.text(
            0.02, 0.98,
            f'Convergence Delay: {delay} rounds',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_summary_report(
    results: Dict[str, any],
    save_path: Optional[str] = None
) -> str:
    """
    Create a text summary report of attack results.

    Args:
        results: Dictionary containing experiment results
        save_path: Path to save the report (None to skip saving)

    Returns:
        Summary report as string
    """
    report = []
    report.append("=" * 70)
    report.append("LABEL FLIPPING ATTACK EXPERIMENT SUMMARY")
    report.append("=" * 70)
    report.append("")

    # Attack configuration
    if "attack_config" in results:
        config = results["attack_config"]
        report.append("Attack Configuration:")
        report.append(f"  Attack Type: {config.get('attack_type', 'N/A')}")
        report.append(f"  Flip Rate: {config.get('flip_rate', 'N/A')}")
        report.append(f"  Malicious Fraction: {config.get('malicious_fraction', 'N/A')}")
        report.append("")

    # Comparison metrics
    if "comparison" in results:
        comp = results["comparison"]
        report.append("Impact Metrics:")
        report.append(f"  Baseline Accuracy: {comp.get('final_accuracy_baseline', 'N/A'):.4f}")
        report.append(f"  Attacked Accuracy: {comp.get('final_accuracy_attacked', 'N/A'):.4f}")
        report.append(f"  Accuracy Drop: {comp.get('accuracy_drop', 'N/A'):.4f}")
        report.append("")

        if comp.get("attack_success_metrics"):
            success = comp["attack_success_metrics"]
            report.append("Attack Success:")
            report.append(f"  Accuracy Degradation: {success.get('accuracy_degradation', 'N/A'):.4f}")
            report.append(f"  Relative Degradation: {success.get('relative_degradation', 'N/A'):.2%}")
            report.append(f"  Attack Successful: {success.get('attack_success', 'N/A')}")
            report.append("")

        if comp.get("convergence_metrics"):
            conv = comp["convergence_metrics"]
            report.append("Convergence Analysis:")
            report.append(f"  Baseline Convergence: Round {conv.get('baseline_convergence_round', 'N/A')}")
            report.append(f"  Attacked Convergence: Round {conv.get('attacked_convergence_round', 'N/A')}")
            report.append(f"  Convergence Delay: {conv.get('convergence_delay', 'N/A')} rounds")
            report.append("")

    report.append("=" * 70)

    report_text = "\n".join(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Saved report to {save_path}")

    return report_text
