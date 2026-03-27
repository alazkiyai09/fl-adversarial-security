"""
Attack Metrics - Evaluate property inference attack performance.

This module provides metrics for assessing how well a meta-classifier
can infer dataset properties.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute regression metrics for property inference.

    Args:
        y_true: True property values
        y_pred: Predicted property values

    Returns:
        Dict with MAE, MSE, RMSE, R²

    Example:
        >>> y_true = np.array([0.05, 0.10, 0.15, 0.20])
        >>> y_pred = np.array([0.06, 0.11, 0.14, 0.19])
        >>> metrics = compute_regression_metrics(y_true, y_pred)
        >>> metrics['MAE']
        0.01
    """
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'MSE': float(mean_squared_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred))
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute classification metrics for property inference.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_prob: Predicted probabilities (optional, for AUC)

    Returns:
        Dict with accuracy, precision, recall, F1, AUC

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0, 1, 1, 1])
        >>> metrics = compute_classification_metrics(y_true, y_pred)
        >>> metrics['accuracy']
        0.75
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }

    if y_prob is not None:
        try:
            metrics['auc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics['auc'] = 0.5

    return metrics


def compute_confidence_intervals(
    metrics: Dict[str, float],
    n_samples: int,
    confidence: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """Compute confidence intervals for metrics via bootstrap.

    Args:
        metrics: Dict of metric values
        n_samples: Number of samples (for CI calculation)
        confidence: Confidence level (e.g., 0.95)

    Returns:
        Dict mapping metric names to (lower, upper) tuples

    Example:
        >>> metrics = {'MAE': 0.05, 'R2': 0.80}
        >>> cis = compute_confidence_intervals(metrics, n_samples=100)
        >>> cis['MAE']
        (0.03, 0.07)
    """
    # For simplicity, use normal approximation
    # In practice, you'd use bootstrap resampling
    z_score = stats.norm.ppf((1 + confidence) / 2)

    intervals = {}
    for key, value in metrics.items():
        # Assume standard error scales with 1/sqrt(n)
        se = value / np.sqrt(n_samples)
        margin = z_score * se
        intervals[key] = (max(0, value - margin), value + margin)

    return intervals


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Compute confidence interval via bootstrap resampling.

    Args:
        y_true: True values
        y_pred: Predicted values
        metric_func: Function to compute metric (e.g., mean_absolute_error)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level

    Returns:
        (mean, lower_bound, upper_bound) tuple

    Example:
        >>> y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> y_pred = np.array([0.12, 0.18, 0.31, 0.39, 0.51])
        >>> mean, lower, upper = bootstrap_confidence_interval(
        ...     y_true, y_pred, mean_absolute_error, n_bootstrap=1000
        ... )
    """
    np.random.seed(42)
    n = len(y_true)
    boot_metrics = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Compute metric
        metric = metric_func(y_true_boot, y_pred_boot)
        boot_metrics.append(metric)

    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(boot_metrics, 100 * alpha / 2)
    upper = np.percentile(boot_metrics, 100 * (1 - alpha / 2))
    mean = np.mean(boot_metrics)

    return float(mean), float(lower), float(upper)


def compare_to_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compare attack performance to naive baseline.

    Args:
        y_true: True property values
        y_pred: Attack predictions
        baseline_pred: Baseline predictions (None for mean/majority)

    Returns:
        Dict with comparison metrics

    Example:
        >>> y_true = np.array([0.05, 0.10, 0.15])
        >>> y_pred = np.array([0.06, 0.11, 0.14])
        >>> result = compare_to_baseline(y_true, y_pred)
        >>> result['attack_better']
        True
    """
    # Compute baseline prediction (mean of true values)
    if baseline_pred is None:
        baseline_pred = np.full_like(y_true, y_true.mean())

    # Compute metrics
    attack_mae = mean_absolute_error(y_true, y_pred)
    baseline_mae = mean_absolute_error(y_true, baseline_pred)

    attack_r2 = r2_score(y_true, y_pred)
    baseline_r2 = r2_score(y_true, baseline_pred)

    return {
        'attack_MAE': float(attack_mae),
        'baseline_MAE': float(baseline_mae),
        'MAE_improvement': float(baseline_mae - attack_mae),
        'MAE_ratio': float(attack_mae / (baseline_mae + 1e-8)),
        'attack_R2': float(attack_r2),
        'baseline_R2': float(baseline_r2),
        'R2_improvement': float(attack_r2 - baseline_r2),
        'attack_better': attack_mae < baseline_mae
    }


def compute_rank_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute rank correlation metrics.

    Useful for assessing whether attack correctly ranks clients by property.

    Args:
        y_true: True property values
        y_pred: Predicted property values

    Returns:
        Dict with Spearman and Kendall correlations

    Example:
        >>> y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> y_pred = np.array([0.12, 0.18, 0.31, 0.39, 0.51])
        >>> corr = compute_rank_correlation(y_true, y_pred)
        >>> corr['spearman'] > 0.9
        True
    """
    spearman, _ = stats.spearmanr(y_true, y_pred)
    kendall, _ = stats.kendalltau(y_true, y_pred)

    return {
        'spearman': float(spearman),
        'kendall': float(kendall)
    }


def compute_error_by_property_range(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 5
) -> Dict[str, Dict[str, float]]:
    """Compute error metrics binned by property value.

    Useful for understanding which property ranges are easier/harder to infer.

    Args:
        y_true: True property values
        y_pred: Predicted property values
        n_bins: Number of bins to create

    Returns:
        Dict mapping bin ranges to error metrics

    Example:
        >>> y_true = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        >>> y_pred = np.array([0.06, 0.11, 0.14, 0.19, 0.26])
        >>> binned = compute_error_by_property_range(y_true, y_pred, n_bins=3)
    """
    # Create bins
    bins = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    bin_indices = np.digitize(y_true, bins[:-1])

    results = {}

    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_true = y_true[mask]
            bin_pred = y_pred[mask]

            bin_range = f"({bins[i-1]:.3f}, {bins[i]:.3f}]"
            results[bin_range] = {
                'MAE': float(mean_absolute_error(bin_true, bin_pred)),
                'RMSE': float(np.sqrt(mean_squared_error(bin_true, bin_pred))),
                'n_samples': int(mask.sum())
            }

    return results


def compute_attack_success_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, float]:
    """Compute success rate based on error threshold.

    Args:
        y_true: True property values
        y_pred: Predicted property values
        threshold: Maximum allowable absolute error

    Returns:
        Dict with success rate metrics

    Example:
        >>> y_true = np.array([0.10, 0.15, 0.20])
        >>> y_pred = np.array([0.11, 0.16, 0.21])
        >>> success = compute_attack_success_rate(y_true, y_pred, threshold=0.05)
        >>> success['within_threshold']
        0.333
    """
    errors = np.abs(y_true - y_pred)
    within_threshold = (errors <= threshold).mean()

    return {
        'within_threshold': float(within_threshold),
        'mean_error': float(errors.mean()),
        'std_error': float(errors.std()),
        'median_error': float(np.median(errors))
    }


def evaluate_multi_property_attack(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    property_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Evaluate attack on multiple properties simultaneously.

    Args:
        y_true: True property values of shape (n_samples, n_properties)
        y_pred: Predicted property values of shape (n_samples, n_properties)
        property_names: Names of each property

    Returns:
        Nested dict with metrics for each property

    Example:
        >>> y_true = np.array([[0.1, 100], [0.2, 200]])
        >>> y_pred = np.array([[0.11, 110], [0.19, 190]])
        >>> results = evaluate_multi_property_attack(
        ...     y_true, y_pred, ['fraud_rate', 'dataset_size']
        ... )
        >>> results['fraud_rate']['MAE']
        0.01
    """
    results = {}

    for i, prop_name in enumerate(property_names):
        prop_true = y_true[:, i]
        prop_pred = y_pred[:, i]

        results[prop_name] = compute_regression_metrics(prop_true, prop_pred)

    return results


def compute_temporal_metrics(
    y_true_by_round: List[np.ndarray],
    y_pred_by_round: List[np.ndarray]
) -> Dict[str, Any]:
    """Compute attack performance across FL rounds.

    Args:
        y_true_by_round: True values for each round
        y_pred_by_round: Predicted values for each round

    Returns:
        Dict with temporal metrics

    Example:
        >>> y_true_by_round = [np.array([0.1, 0.2]), np.array([0.15, 0.25])]
        >>> y_pred_by_round = [np.array([0.11, 0.21]), np.array([0.16, 0.24])]
        >>> temporal = compute_temporal_metrics(y_true_by_round, y_pred_by_round)
    """
    round_maes = []
    round_r2s = []

    for y_true, y_pred in zip(y_true_by_round, y_pred_by_round):
        round_maes.append(mean_absolute_error(y_true, y_pred))
        round_r2s.append(r2_score(y_true, y_pred))

    return {
        'MAE_by_round': [float(mae) for mae in round_maes],
        'R2_by_round': [float(r2) for r2 in round_r2s],
        'mean_MAE': float(np.mean(round_maes)),
        'std_MAE': float(np.std(round_maes)),
        'MAE_improvement_early_vs_late': float(round_maes[-1] - round_maes[0])
    }
