"""
Evaluation metrics for FL anomaly detection.
Computes precision, recall, F1, FPR, and generates ROC curves.
"""

from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix
)


def compute_detection_metrics(
    predictions: List[bool],
    ground_truth: List[bool]
) -> Dict[str, float]:
    """
    Compute detection metrics given predictions and ground truth.

    Args:
        predictions: List of binary predictions (True = malicious)
        ground_truth: List of ground truth labels (True = malicious)

    Returns:
        Dictionary with metrics:
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1: 2 * (precision * recall) / (precision + recall)
        - fpr: FP / (FP + TN) (false positive rate)
        - fnr: FN / (FN + TP) (false negative rate)
        - tpr: TP / (TP + FN) (true positive rate, same as recall)
        - accuracy: (TP + TN) / (TP + TN + FP + FN)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have same length")

    # Convert to numpy arrays
    y_pred = np.array(predictions)
    y_true = np.array(ground_truth)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = recall
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'tpr': float(tpr),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def compute_detection_latency(
    predictions_per_round: List[List[bool]],
    ground_truth: List[bool]
) -> Dict[str, any]:
    """
    Compute detection latency (rounds to detect each malicious client).

    Args:
        predictions_per_round: List of predictions for each round
                               [[round_1_preds], [round_2_preds], ...]
        ground_truth: Ground truth labels (index = client_id)

    Returns:
        Dictionary with latency statistics:
        - latencies: List of rounds to detect each malicious client
        - mean_latency: Average rounds to detection
        - max_latency: Maximum rounds to detection
        - detected_count: Number of malicious clients detected
        - total_malicious: Total number of malicious clients
        - detection_rate: Fraction of malicious clients detected
    """
    # Find malicious client indices
    malicious_indices = [
        i for i, is_malicious in enumerate(ground_truth)
        if is_malicious
    ]

    if not malicious_indices:
        return {
            'latencies': [],
            'mean_latency': 0,
            'max_latency': 0,
            'detected_count': 0,
            'total_malicious': 0,
            'detection_rate': 0.0
        }

    latencies = []

    for client_idx in malicious_indices:
        detected_round = None

        for round_num, round_preds in enumerate(predictions_per_round):
            if round_num < len(round_preds) and round_preds[client_idx]:
                detected_round = round_num + 1  # 1-indexed
                break

        if detected_round is not None:
            latencies.append(detected_round)

    return {
        'latencies': latencies,
        'mean_latency': float(np.mean(latencies)) if latencies else float('inf'),
        'max_latency': float(np.max(latencies)) if latencies else float('inf'),
        'detected_count': len(latencies),
        'total_malicious': len(malicious_indices),
        'detection_rate': len(latencies) / len(malicious_indices)
    }


def plot_roc_curve(
    anomaly_scores: List[float],
    ground_truth: List[bool],
    save_path: str = None,
    show_plot: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Plot ROC curve and compute AUC.

    Args:
        anomaly_scores: List of anomaly scores (higher = more anomalous)
        ground_truth: Ground truth labels (True = malicious)
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot

    Returns:
        Tuple of (fpr, tpr, auc_score)
    """
    fpr, tpr, thresholds = roc_curve(ground_truth, anomaly_scores)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve for Anomaly Detection')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fpr, tpr, auc_score


def plot_precision_recall_curve(
    anomaly_scores: List[float],
    ground_truth: List[bool],
    save_path: str = None,
    show_plot: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Plot precision-recall curve and compute average precision.

    Args:
        anomaly_scores: List of anomaly scores
        ground_truth: Ground truth labels
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot

    Returns:
        Tuple of (precision, recall, avg_precision)
    """
    precision, recall, thresholds = precision_recall_curve(ground_truth, anomaly_scores)
    avg_precision = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Anomaly Detection')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return precision, recall, avg_precision


def find_optimal_threshold(
    anomaly_scores: List[float],
    ground_truth: List[bool],
    metric: str = "f1"
) -> float:
    """
    Find optimal threshold for binary decision.

    Args:
        anomaly_scores: List of anomaly scores
        ground_truth: Ground truth labels
        metric: Metric to optimize ("f1", "accuracy", "balanced_accuracy")

    Returns:
        Optimal threshold value
    """
    scores = np.array(anomaly_scores)
    labels = np.array(ground_truth)

    # Try different thresholds
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best_threshold = 0.5
    best_score = -1

    for threshold in thresholds:
        predictions = scores > threshold

        if metric == "f1":
            metrics = compute_detection_metrics(predictions.tolist(), labels.tolist())
            score = metrics['f1']
        elif metric == "accuracy":
            metrics = compute_detection_metrics(predictions.tolist(), labels.tolist())
            score = metrics['accuracy']
        elif metric == "balanced_accuracy":
            metrics = compute_detection_metrics(predictions.tolist(), labels.tolist())
            # Balanced accuracy = (tpr + tnr) / 2
            tnr = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
            score = (metrics['tpr'] + tnr) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def compare_detectors(
    detector_scores: Dict[str, List[float]],
    ground_truth: List[bool]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple detectors by computing metrics for each.

    Args:
        detector_scores: Dictionary mapping detector_name -> list of scores
        ground_truth: Ground truth labels

    Returns:
        Dictionary mapping detector_name -> metrics dictionary
    """
    results = {}

    for detector_name, scores in detector_scores.items():
        # Find optimal threshold for this detector
        threshold = find_optimal_threshold(scores, ground_truth, metric="f1")

        # Make predictions
        predictions = [score > threshold for score in scores]

        # Compute metrics
        metrics = compute_detection_metrics(predictions, ground_truth)
        metrics['threshold_used'] = threshold

        results[detector_name] = metrics

    return results
