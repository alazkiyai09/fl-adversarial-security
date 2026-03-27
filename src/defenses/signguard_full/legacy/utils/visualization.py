"""Visualization utilities for SignGuard experiments."""

import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Make matplotlib optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    np = None


if HAS_MATPLOTLIB:
    # Set publication-style plotting parameters
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (6, 4),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_reputation_evolution(
    reputation_history: Dict[str, List[float]],
    honest_clients: List[str],
    malicious_clients: List[str],
    output_path: Optional[str] = None,
    show_std: bool = False,
):
    """Plot reputation evolution over FL rounds.

    Args:
        reputation_history: Client ID -> list of reputation values
        honest_clients: List of honest client IDs
        malicious_clients: List of malicious client IDs
        output_path: Optional path to save figure
        show_std: Whether to show standard deviation bands

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    rounds = list(range(len(next(iter(reputation_history.values())))))
    
    # Plot honest clients (blue)
    for client_id in honest_clients:
        if client_id in reputation_history:
            ax.plot(
                rounds,
                reputation_history[client_id],
                color='steelblue',
                alpha=0.6,
                linewidth=1,
            )
    
    # Plot average honest reputation
    honest_reps = np.array([
        reputation_history[c] for c in honest_clients if c in reputation_history
    ])
    if len(honest_reps) > 0:
        mean_honest = np.mean(honest_reps, axis=0)
        ax.plot(
            rounds,
            mean_honest,
            color='darkblue',
            linewidth=2.5,
            label='Honest Clients',
        )
        if show_std:
            std_honest = np.std(honest_reps, axis=0)
            ax.fill_between(
                rounds,
                mean_honest - std_honest,
                mean_honest + std_honest,
                color='steelblue',
                alpha=0.2,
            )
    
    # Plot malicious clients (red)
    for client_id in malicious_clients:
        if client_id in reputation_history:
            ax.plot(
                rounds,
                reputation_history[client_id],
                color='crimson',
                alpha=0.6,
                linewidth=1,
                linestyle='--',
            )
    
    # Plot average malicious reputation
    mal_reps = np.array([
        reputation_history[c] for c in malicious_clients if c in reputation_history
    ])
    if len(mal_reps) > 0:
        mean_mal = np.mean(mal_reps, axis=0)
        ax.plot(
            rounds,
            mean_mal,
            color='darkred',
            linewidth=2.5,
            linestyle='--',
            label='Malicious Clients',
        )
        if show_std:
            std_mal = np.std(mal_reps, axis=0)
            ax.fill_between(
                rounds,
                mean_mal - std_mal,
                mean_mal + std_mal,
                color='crimson',
                alpha=0.2,
            )
    
    ax.set_xlabel('Federated Learning Round')
    ax.set_ylabel('Reputation Score')
    ax.set_title('Client Reputation Evolution')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best')
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def plot_detection_roc(
    true_positive_rates: Dict[str, List[float]],
    false_positive_rates: Dict[str, List[float]],
    auc_scores: Dict[str, float],
    output_path: Optional[str] = None,
):
    """Plot ROC curves for anomaly detection.

    Args:
        true_positive_rates: Method name -> list of TPR values
        false_positive_rates: Method name -> list of FPR values
        auc_scores: Method name -> AUC score
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    
    # Colors for different methods
    colors = {
        'SignGuard': 'darkblue',
        'FoolsGold': 'darkgreen',
        'Krum': 'darkorange',
        'Magnitude': 'purple',
        'Direction': 'brown',
    }
    
    # Plot ROC curves
    for method, tpr_list in true_positive_rates.items():
        fpr_list = false_positive_rates[method]
        auc = auc_scores.get(method, 0.0)
        
        color = colors.get(method, None)
        label = f'{method} (AUC = {auc:.3f})'
        
        ax.plot(
            fpr_list,
            tpr_list,
            color=color,
            linewidth=2,
            label=label,
        )
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Anomaly Detection ROC Curves')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def plot_privacy_utility(
    epsilon_values: List[float],
    accuracy_values: Dict[str, List[float]],
    asr_values: Dict[str, List[float]],
    output_path: Optional[str] = None,
):
    """Plot privacy-utility trade-off with DP.

    Args:
        epsilon_values: List of epsilon values for DP
        accuracy_values: Defense name -> list of accuracy values
        asr_values: Defense name -> list of ASR values
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    colors = {
        'SignGuard': 'darkblue',
        'SignGuard+DP': 'darkgreen',
        'FedAvg': 'gray',
    }
    
    # Plot accuracy vs epsilon
    for defense, acc_values in accuracy_values.items():
        color = colors.get(defense, None)
        ax1.plot(
            epsilon_values,
            acc_values,
            marker='o',
            color=color,
            linewidth=2,
            label=defense,
        )
    
    ax1.set_xlabel('Privacy Budget (ε)')
    ax1.set_ylabel('Model Accuracy')
    ax1.set_title('Accuracy vs Privacy')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot ASR vs epsilon
    for defense, asr_vals in asr_values.items():
        color = colors.get(defense, None)
        ax2.plot(
            epsilon_values,
            asr_vals,
            marker='s',
            color=color,
            linewidth=2,
            label=defense,
        )
    
    ax2.set_xlabel('Privacy Budget (ε)')
    ax2.set_ylabel('Attack Success Rate')
    ax2.set_title('Attack Success Rate vs Privacy')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def plot_defense_comparison(
    defense_names: List[str],
    accuracy_values: Dict[str, List[float]],
    attack_types: List[str],
    output_path: Optional[str] = None,
    y_label: str = 'Accuracy',
):
    """Plot grouped bar chart for defense comparison.

    Args:
        defense_names: List of defense names
        accuracy_values: Attack type -> list of accuracy values
        attack_types: List of attack types (for x-axis)
        output_path: Optional path to save figure
        y_label: Y-axis label

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(attack_types))
    width = 0.8 / len(defense_names)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(defense_names)))
    
    for i, defense in enumerate(defense_names):
        offset = (i - len(defense_names) / 2 + 0.5) * width
        values = accuracy_values.get(defense, [0] * len(attack_types))
        
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=defense,
            color=colors[i],
            alpha=0.8,
        )
    
    ax.set_xlabel('Attack Type')
    ax.set_ylabel(y_label)
    ax.set_title('Defense Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def plot_ablation_study(
    component_names: List[str],
    accuracy_values: List[float],
    asr_values: List[float],
    output_path: Optional[str] = None,
):
    """Plot ablation study results.

    Args:
        component_names: List of component configurations
        accuracy_values: Accuracy for each configuration
        asr_values: ASR for each configuration
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    x = np.arange(len(component_names))
    width = 0.35
    
    # Plot accuracy
    bars1 = ax1.bar(
        x - width / 2,
        accuracy_values,
        width,
        label='Accuracy',
        color='steelblue',
        alpha=0.8,
    )
    
    # Plot ASR on twin axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        asr_values,
        width,
        label='Attack Success Rate',
        color='crimson',
        alpha=0.8,
    )
    
    ax1.set_xlabel('SignGuard Components')
    ax1.set_ylabel('Accuracy', color='steelblue')
    ax2.set_ylabel('Attack Success Rate', color='crimson')
    ax1.set_title('Ablation Study: Component Contribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(component_names, rotation=15, ha='right')
    
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='crimson')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    ax1.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def plot_training_progress(
    rounds: List[int],
    train_metrics: Dict[str, List[float]],
    test_metrics: Dict[str, List[float]],
    output_path: Optional[str] = None,
):
    """Plot training progress over rounds.

    Args:
        rounds: List of round numbers
        train_metrics: Metric name -> list of training values
        test_metrics: Metric name -> list of test values
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot accuracy
    ax1 = axes[0]
    if 'accuracy' in train_metrics:
        ax1.plot(
            rounds,
            train_metrics['accuracy'],
            label='Train',
            marker='o',
            color='steelblue',
        )
    if 'accuracy' in test_metrics:
        ax1.plot(
            rounds,
            test_metrics['accuracy'],
            label='Test',
            marker='s',
            color='darkgreen',
        )
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2 = axes[1]
    if 'loss' in train_metrics:
        ax2.plot(
            rounds,
            train_metrics['loss'],
            label='Train',
            marker='o',
            color='steelblue',
        )
    if 'loss' in test_metrics:
        ax2.plot(
            rounds,
            test_metrics['loss'],
            label='Test',
            marker='s',
            color='darkgreen',
        )
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def plot_overhead_comparison(
    methods: List[str],
    time_overhead: Dict[str, float],
    comm_overhead: Dict[str, float],
    memory_overhead: Dict[str, float],
    output_path: Optional[str] = None,
):
    """Plot computational overhead comparison.

    Args:
        methods: List of defense methods
        time_overhead: Method -> time overhead (ms)
        comm_overhead: Method -> communication overhead (MB)
        memory_overhead: Method -> memory overhead (MB)
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    x = np.arange(len(methods))
    
    # Plot time overhead
    axes[0].bar(x, list(time_overhead.values()), color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Time per Round (ms)')
    axes[0].set_title('Execution Time Overhead')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot communication overhead
    axes[1].bar(x, list(comm_overhead.values()), color='darkgreen', alpha=0.8)
    axes[1].set_ylabel('Communication (MB)')
    axes[1].set_title('Communication Overhead')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot memory overhead
    axes[2].bar(x, list(memory_overhead.values()), color='darkorange', alpha=0.8)
    axes[2].set_ylabel('Memory (MB)')
    axes[2].set_title('Memory Overhead')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=15, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path)
    
    return fig


def save_all_figures(
    figures: Dict[str, 'plt.Figure'],
    output_dir: str,
    format: str = 'pdf',
):
    """Save all figures to directory.

    Args:
        figures: Dictionary mapping figure name to Figure object
        output_dir: Directory to save figures
        format: File format ('pdf', 'png', 'svg')
    """
    _check_matplotlib()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, fig in figures.items():
        file_path = output_path / f"{name}.{format}"
        fig.savefig(str(file_path), format=format, bbox_inches='tight')
        plt.close(fig)


def create_table_from_results(
    results: Dict[str, Dict[str, float]],
    caption: str,
    label: str,
    output_path: Optional[str] = None,
) -> str:
    """Create LaTeX table from results dictionary.

    Args:
        results: Nested dictionary (row -> col -> value)
        caption: Table caption
        label: Table label for LaTeX reference
        output_path: Optional path to save table

    Returns:
        LaTeX table string
    """
    rows = sorted(results.keys())
    cols = sorted(set().union(*[results[r].keys() for r in rows]))
    
    latex = []
    latex.append(f"\\begin{{table}}[h]")
    latex.append(f"\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(f"\\begin{{tabular}}{{{'l ' + 'c ' * (len(cols) - 1)}}}")
    latex.append(f"\\toprule")
    
    # Header row
    latex.append(" & " + " & ".join(cols) + " \\\\")
    latex.append("\\midrule")
    
    # Data rows
    for row in rows:
        values = [results[row].get(col, 0.0) for col in cols]
        formatted_values = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in values]
        latex.append(f"{row} & " + " & ".join(formatted_values) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    table_str = "\n".join(latex)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table_str)
    
    return table_str
