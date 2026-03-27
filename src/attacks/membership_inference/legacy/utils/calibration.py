"""
Threshold Calibration for Membership Inference Attacks

This module provides utilities to calibrate attack thresholds
to achieve specific false positive rates or optimize attack metrics.
"""

import numpy as np
from sklearn.metrics import roc_curve
from typing import Tuple


def calibrate_threshold_on_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.05
) -> float:
    """
    Calibrate threshold to achieve target false positive rate.

    Args:
        scores: Attack scores (higher = more likely member)
        labels: True labels (1 = member, 0 = non-member)
        target_fpr: Desired false positive rate

    Returns:
        Threshold that achieves (approximately) target FPR
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find threshold closest to target FPR
    idx = np.argmin(np.abs(fpr - target_fpr))

    if idx == 0:
        # If we can't achieve the target FPR, use minimum threshold
        threshold = np.min(scores) - 1.0
    else:
        threshold = thresholds[idx]

    return float(threshold)


def calibrate_threshold_on_tpr(
    scores: np.ndarray,
    labels: np.ndarray,
    target_tpr: float = 0.95
) -> float:
    """
    Calibrate threshold to achieve target true positive rate.

    Args:
        scores: Attack scores (higher = more likely member)
        labels: True labels (1 = member, 0 = non-member)
        target_tpr: Desired true positive rate

    Returns:
        Threshold that achieves (approximately) target TPR
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find threshold closest to target TPR
    idx = np.argmin(np.abs(tpr - target_tpr))

    if idx == 0:
        threshold = np.min(scores) - 1.0
    else:
        threshold = thresholds[idx]

    return float(threshold)


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    optimization_metric: str = 'youden'
) -> Tuple[float, float]:
    """
    Find optimal threshold using specified optimization metric.

    Args:
        scores: Attack scores (higher = more likely member)
        labels: True labels (1 = member, 0 = non-member)
        optimization_metric: Metric to optimize ('youden', 'accuracy', 'f1')

    Returns:
        (optimal_threshold, metric_value)
    """
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

    if optimization_metric == 'youden':
        # Youden's index: TPR - FPR
        fpr, tpr, thresholds = roc_curve(labels, scores)
        youden_index = tpr - fpr
        idx = np.argmax(youden_index)
        optimal_threshold = thresholds[idx]
        metric_value = youden_index[idx]

    elif optimization_metric == 'accuracy':
        # Brute force search for best accuracy
        best_threshold = 0.0
        best_accuracy = 0.0

        for threshold in np.percentile(scores, np.arange(0, 101, 1)):
            predictions = (scores >= threshold).astype(int)
            acc = accuracy_score(labels, predictions)

            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = threshold

        optimal_threshold = best_threshold
        metric_value = best_accuracy

    elif optimization_metric == 'f1':
        # Brute force search for best F1 score
        best_threshold = 0.0
        best_f1 = 0.0

        for threshold in np.percentile(scores, np.arange(0, 101, 1)):
            predictions = (scores >= threshold).astype(int)
            f1 = f1_score(labels, predictions, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_threshold = best_threshold
        metric_value = best_f1

    else:
        raise ValueError(f"Unknown optimization metric: {optimization_metric}")

    return float(optimal_threshold), float(metric_value)


def compute_threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> dict:
    """
    Compute attack metrics at a specific threshold.

    Args:
        scores: Attack scores (higher = more likely member)
        labels: True labels (1 = member, 0 = non-member)
        threshold: Decision threshold

    Returns:
        Dictionary of metrics at threshold
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    metrics = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(labels, predictions)),
        'true_positive_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'true_negative_rate': float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'false_negative_rate': float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }

    return metrics


def analyze_threshold_sensitivity(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold_range: Tuple[float, float] = (0.0, 1.0),
    n_steps: int = 100
) -> dict:
    """
    Analyze attack performance across a range of thresholds.

    Args:
        scores: Attack scores (higher = more likely member)
        labels: True labels (1 = member, 0 = non-member)
        threshold_range: (min, max) threshold values to test
        n_steps: Number of thresholds to test

    Returns:
        Dictionary with arrays of metrics across thresholds
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)

    tprs = []
    fprs = []
    accuracies = []
    precisions = []
    recalls = []

    for threshold in thresholds:
        metrics = compute_threshold_metrics(scores, labels, threshold)

        tprs.append(metrics['true_positive_rate'])
        fprs.append(metrics['false_positive_rate'])
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])

    return {
        'thresholds': thresholds,
        'tprs': np.array(tprs),
        'fprs': np.array(fprs),
        'accuracies': np.array(accuracies),
        'precisions': np.array(precisions),
        'recalls': np.array(recalls)
    }


if __name__ == "__main__":
    print("This module provides threshold calibration utilities.")
    print("Import and use functions in your attack scripts.")
