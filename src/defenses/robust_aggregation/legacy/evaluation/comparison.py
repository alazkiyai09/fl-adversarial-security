"""
Comparison matrix and visualization for robust aggregator evaluation.

This module provides tools to compare multiple aggregators across
different attack types and attacker fractions.
"""

from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_comparison_matrix(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = 'accuracy'
) -> pd.DataFrame:
    """
    Generate comparison matrix for aggregators × attacks × fractions.

    Args:
        results: Nested dict structure:
                results[aggregator][attack][fraction] = metric_value
                Example:
                {
                    'Median': {
                        'label_flipping': {'0.1': 0.95, '0.2': 0.92},
                        'backdoor': {'0.1': 0.93, '0.2': 0.88}
                    },
                    'Krum': {...}
                }
        metric: Metric to display ('accuracy', 'asr', 'convergence_speed')

    Returns:
        DataFrame with multi-index columns (attack, fraction)
        and rows as aggregators

    Example output:
                  label_flipping              backdoor
                        0.1    0.2    0.3     0.1    0.2    0.3
        Median        0.95   0.92   0.85    0.93   0.88   0.80
        Krum          0.94   0.90   0.82    0.92   0.85   0.75
    """
    # Get all unique values
    aggregators = sorted(results.keys())

    if not aggregators:
        return pd.DataFrame()

    attacks = set()
    fractions = set()

    for agg_results in results.values():
        for attack_name in agg_results.keys():
            attacks.add(attack_name)
            for fraction in agg_results[attack_name].keys():
                fractions.add(fraction)

    attacks = sorted(attacks)
    fractions = sorted(fractions, key=lambda x: float(x))

    # Build matrix
    data = []
    for agg in aggregators:
        row = {}
        for attack in attacks:
            for fraction in fractions:
                # Get value if exists, otherwise NaN
                value = results.get(agg, {}).get(attack, {}).get(fraction, np.nan)
                col_name = (attack, fraction)
                row[col_name] = value
        data.append(row)

    # Create DataFrame with multi-index columns
    df = pd.DataFrame(data, index=aggregators)
    df.columns = pd.MultiIndex.from_tuples(
        df.columns,
        names=['Attack', 'Attacker Fraction']
    )

    return df


def generate_heatmap(
    matrix: pd.DataFrame,
    metric: str = 'accuracy',
    title: str = None,
    cmap: str = 'RdYlGn',
    figsize: tuple = (12, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Generate heatmap visualization of comparison matrix.

    Args:
        matrix: Comparison matrix DataFrame from generate_comparison_matrix()
        metric: Metric being displayed (for title and colorbar)
        title: Optional custom title
        cmap: Colormap name ('RdYlGn' = red-yellow-green for accuracy)
              For ASR, use 'RdYlGn_r' (reversed, so red=high ASR=bad)
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    if matrix.empty:
        print("Warning: Empty matrix, cannot generate heatmap")
        return None

    # Set style
    sns.set_style("whitegrid")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colormap based on metric
    if metric == 'asr':
        # For ASR: lower is better, so reverse colormap
        cmap = 'RdYlGn_r'
        vmin, vmax = 0, 1
    elif metric == 'accuracy':
        # For accuracy: higher is better
        cmap = 'RdYlGn'
        vmin, vmax = 0, 1
    elif metric == 'convergence_speed':
        # For convergence: lower is better (fewer rounds)
        cmap = 'RdYlGn_r'
        vmin, vmax = 0, matrix.max().max()
    else:
        cmap = 'viridis'
        vmin, vmax = None, None

    # Create heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={'label': metric.replace('_', ' ').title()},
        ax=ax
    )

    # Set title
    if title is None:
        title = f'Aggregator Comparison: {metric.replace("_", " ").title()}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def generate_summary_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = 'accuracy'
) -> pd.DataFrame:
    """
    Generate summary table averaging across attacker fractions.

    Args:
        results: Nested dict as in generate_comparison_matrix()
        metric: Metric to summarize

    Returns:
        DataFrame with aggregators as rows, attacks as columns,
        values averaged across all fractions
    """
    matrix = generate_comparison_matrix(results, metric)

    if matrix.empty:
        return pd.DataFrame()

    # Average across fractions for each attack
    summary = matrix.groupby(level=0, axis=1).mean()

    return summary


def plot_aggregator_performance(
    results: Dict[str, Dict[str, Dict[str, float]]],
    aggregator_names: List[str],
    metric: str = 'accuracy',
    figsize: tuple = (10, 6),
    save_path: str = None
) -> plt.Figure:
    """
    Plot line graph comparing aggregator performance vs attacker fraction.

    Args:
        results: Nested dict as in generate_comparison_matrix()
        aggregator_names: List of aggregator names to plot
        metric: Metric to plot
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique attacks and fractions
    attacks = set()
    fractions = set()

    for agg_results in results.values():
        for attack_name in agg_results.keys():
            attacks.add(attack_name)
            for fraction in agg_results[attack_name].keys():
                fractions.add(fraction)

    attacks = sorted(attacks)
    fractions = sorted(fractions, key=lambda x: float(x))

    # Create subplots for each attack type
    n_attacks = len(attacks)
    fig, axes = plt.subplots(1, n_attacks, figsize=(figsize[0] * n_attacks, figsize[1]))

    if n_attacks == 1:
        axes = [axes]

    for idx, attack in enumerate(attacks):
        ax = axes[idx]

        # Plot line for each aggregator
        for agg_name in aggregator_names:
            if agg_name not in results:
                continue

            agg_data = results[agg_name].get(attack, {})
            x_values = []
            y_values = []

            for fraction in fractions:
                if fraction in agg_data:
                    x_values.append(float(fraction))
                    y_values.append(agg_data[fraction])

            if x_values:
                ax.plot(x_values, y_values, marker='o', label=agg_name)

        ax.set_xlabel('Attacker Fraction')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{attack.replace("_", " ").title()} Attack')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def rank_aggregators(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = 'accuracy',
    higher_is_better: bool = True
) -> pd.DataFrame:
    """
    Rank aggregators by average performance across all attacks and fractions.

    Args:
        results: Nested dict as in generate_comparison_matrix()
        metric: Metric to rank by
        higher_is_better: If True, higher metric values are better

    Returns:
        DataFrame with aggregators ranked by performance
    """
    matrix = generate_comparison_matrix(results, metric)

    if matrix.empty:
        return pd.DataFrame()

    # Compute average score for each aggregator
    avg_scores = matrix.mean(axis=1)

    # Sort by score
    if higher_is_better:
        avg_scores = avg_scores.sort_values(ascending=False)
    else:
        avg_scores = avg_scores.sort_values(ascending=True)

    # Create DataFrame
    ranking = pd.DataFrame({
        'Rank': range(1, len(avg_scores) + 1),
        'Aggregator': avg_scores.index,
        f'Average {metric}': avg_scores.values
    })

    ranking = ranking.set_index('Rank')

    return ranking
