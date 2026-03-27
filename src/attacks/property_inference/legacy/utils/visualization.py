"""
Visualization utilities for property inference attack results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import pandas as pd


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    property_name: str = "Property",
    save_path: Optional[str] = None
) -> None:
    """Plot predicted vs actual property values.

    Args:
        y_true: True property values
        y_pred: Predicted property values
        property_name: Name of the property
        save_path: Optional path to save figure

    Example:
        >>> plot_predicted_vs_actual(
        ...     y_true=np.array([0.1, 0.15, 0.2]),
        ...     y_pred=np.array([0.11, 0.14, 0.19]),
        ...     property_name='Fraud Rate'
        ... )
    """
    plt.figure(figsize=(8, 8))

    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # Labels and title
    plt.xlabel(f'True {property_name}', fontsize=12)
    plt.ylabel(f'Predicted {property_name}', fontsize=12)
    plt.title(f'Predicted vs Actual {property_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add MAE as text
    mae = np.abs(y_true - y_pred).mean()
    plt.text(0.05, 0.95, f'MAE: {mae:.4f}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    property_name: str = "Property",
    save_path: Optional[str] = None
) -> None:
    """Plot distribution of prediction errors.

    Args:
        y_true: True property values
        y_pred: Predicted property values
        property_name: Name of the property
        save_path: Optional path to save figure
    """
    errors = y_pred - y_true

    plt.figure(figsize=(10, 6))

    # Histogram
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)

    # Add vertical lines for mean and std
    plt.axvline(errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.4f}')
    plt.axvline(errors.mean() + errors.std(), color='g', linestyle='--', linewidth=2, label=f'±1 Std: {errors.std():.4f}')
    plt.axvline(errors.mean() - errors.std(), color='g', linestyle='--', linewidth=2)

    plt.xlabel(f'Prediction Error ({property_name})', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Prediction Errors for {property_name}', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_temporal_leakage(
    rounds: List[int],
    mae_values: List[float],
    r2_values: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plot how property leakage changes over FL rounds.

    Args:
        rounds: List of round numbers
        mae_values: Attack MAE for each round
        r2_values: Attack R² for each round
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # MAE over rounds
    ax1.plot(rounds, mae_values, 'o-', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('FL Round', fontsize=12)
    ax1.set_ylabel('Attack MAE', fontsize=12)
    ax1.set_title('Attack Error Over FL Rounds', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # R² over rounds
    ax2.plot(rounds, r2_values, 's-', linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('FL Round', fontsize=12)
    ax2.set_ylabel('Attack R²', fontsize=12)
    ax2.set_title('Attack Accuracy Over FL Rounds', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_property_distribution(
    properties: List[float],
    property_name: str = "Property",
    save_path: Optional[str] = None
) -> None:
    """Plot distribution of a property across clients.

    Args:
        properties: List of property values
        property_name: Name of the property
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))

    plt.hist(properties, bins=20, edgecolor='black', alpha=0.7)

    # Add statistics
    mean_val = np.mean(properties)
    std_val = np.std(properties)

    plt.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(mean_val + std_val, color='g', linestyle='--', linewidth=2, label=f'±1 Std: {std_val:.4f}')
    plt.axvline(mean_val - std_val, color='g', linestyle='--', linewidth=2)

    plt.xlabel(property_name, fontsize=12)
    plt.ylabel('Number of Clients', fontsize=12)
    plt.title(f'Distribution of {property_name} Across Clients', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_per_client_error(
    client_errors: Dict[int, Dict[str, float]],
    property_name: str = "Property",
    save_path: Optional[str] = None
) -> None:
    """Plot prediction error for each individual client.

    Args:
        client_errors: Dict mapping client IDs to error metrics
        property_name: Name of the property
        save_path: Optional path to save figure
    """
    client_ids = sorted(client_errors.keys())
    mae_values = [client_errors[cid]['MAE'] for cid in client_ids]

    plt.figure(figsize=(12, 6))

    plt.bar(range(len(client_ids)), mae_values, color='steelblue', alpha=0.7)

    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel(f'MAE for {property_name}', fontsize=12)
    plt.title(f'Per-Client Prediction Error', fontsize=14)
    plt.xticks(range(len(client_ids)), [f'Client {cid}' for cid in client_ids], rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # Add average line
    avg_mae = np.mean(mae_values)
    plt.axhline(avg_mae, color='r', linestyle='--', linewidth=2, label=f'Average: {avg_mae:.4f}')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_defense_comparison(
    defense_results: Dict[str, Dict[str, float]],
    metric: str = 'MAE',
    save_path: Optional[str] = None
) -> None:
    """Compare different defense mechanisms.

    Args:
        defense_results: Dict mapping defense names to metrics
        metric: Metric to plot ('MAE', 'R2', etc.)
        save_path: Optional path to save figure
    """
    defenses = list(defense_results.keys())
    values = [defense_results[d].get(metric, 0) for d in defenses]

    plt.figure(figsize=(10, 6))

    # Color gradient from red (bad) to green (good)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(defenses)))

    bars = plt.bar(defenses, values, color=colors)

    plt.ylabel(f'{metric} ({"higher is better" if metric == "R2" else "lower is better"})', fontsize=12)
    plt.xlabel('Defense Type', fontsize=12)
    plt.title(f'Property Inference Defense Comparison ({metric})', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot confusion matrix for classification tasks.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        class_names: Names of classes
        save_path: Optional path to save figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """Plot feature importance from meta-classifier.

    Args:
        feature_names: Names of features
        importance_scores: Importance scores
        top_n: Number of top features to show
        save_path: Optional path to save figure
    """
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1][:top_n]

    plt.figure(figsize=(10, 8))

    plt.barh(range(len(indices)), importance_scores[indices], color='steelblue')

    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature (Update Dimension)', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features for Property Inference', fontsize=14)
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
