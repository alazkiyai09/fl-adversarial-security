"""
Metrics computation for model poisoning experiments.

Tracks accuracy, loss, convergence speed, and attack impact.
"""

import numpy as np
import torch
from typing import Dict, List
from torch.utils.data import DataLoader


def compute_metrics(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Compute evaluation metrics on test dataset.

    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        device: Device for computation

    Returns:
        Dictionary with accuracy, loss, and additional metrics
    """
    model.eval()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute metrics
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(test_loader)

    # Compute precision, recall, F1 for fraud class (class 1)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    true_positives = np.sum((all_preds == 1) & (all_targets == 1))
    false_positives = np.sum((all_preds == 1) & (all_targets == 0))
    false_negatives = np.sum((all_preds == 0) & (all_targets == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def track_convergence(
    metrics_history: List[Dict],
    target_accuracy: float = 0.95,
    patience: int = 5
) -> Dict[str, float]:
    """
    Analyze convergence behavior from training history.

    Args:
        metrics_history: List of metrics per round
        target_accuracy: Target accuracy for convergence
        patience: Number of rounds to stay above target

    Returns:
        Dictionary with convergence metrics
    """
    if not metrics_history:
        return {
            "converged": False,
            "convergence_round": None,
            "final_accuracy": 0.0,
            "total_rounds": 0
        }

    accuracies = [m.get("accuracy", 0.0) for m in metrics_history]
    final_accuracy = accuracies[-1]
    total_rounds = len(metrics_history)

    # Find convergence round
    convergence_round = None
    above_threshold_count = 0

    for i, acc in enumerate(accuracies):
        if acc >= target_accuracy:
            above_threshold_count += 1
            if above_threshold_count >= patience:
                convergence_round = i - patience + 1
                break
        else:
            above_threshold_count = 0

    converged = convergence_round is not None

    return {
        "converged": converged,
        "convergence_round": convergence_round,
        "final_accuracy": final_accuracy,
        "total_rounds": total_rounds,
        "convergence_speed": convergence_round if converged else total_rounds
    }


def compute_attack_impact(
    honest_metrics: Dict,
        attacked_metrics: Dict
) -> Dict[str, float]:
    """
    Compute impact of attack on model performance.

    Args:
        honest_metrics: Metrics without attack
        attacked_metrics: Metrics with attack

    Returns:
        Dictionary with impact metrics
    """
    impact = {}

    # Accuracy degradation
    accuracy_drop = honest_metrics.get("final_accuracy", 0.0) - attacked_metrics.get("final_accuracy", 0.0)
    impact["accuracy_drop"] = accuracy_drop

    # Convergence delay
    honest_conv_round = honest_metrics.get("convergence_round", honest_metrics.get("total_rounds", 0))
    attacked_conv_round = attacked_metrics.get("convergence_round", attacked_metrics.get("total_rounds", 0))

    if honest_metrics.get("converged", False) and not attacked_metrics.get("converged", False):
        impact["convergence_delay"] = float("inf")  # Never converged
    elif not honest_metrics.get("converged", False) and not attacked_metrics.get("converged", False):
        impact["convergence_delay"] = 0  # Neither converged
    else:
        impact["convergence_delay"] = attacked_conv_round - honest_conv_round

    # F1 score degradation
    f1_drop = honest_metrics.get("f1_score", 0.0) - attacked_metrics.get("f1_score", 0.0)
    impact["f1_drop"] = f1_drop

    return impact


def compute_detectability(
    update_history: List[Dict],
    detector_results: List[Dict]
) -> Dict[str, float]:
    """
    Compute detectability metrics for attacks.

    Args:
        update_history: History of client updates
        detector_results: Detection results per round

    Returns:
        Dictionary with detectability metrics
    """
    if not update_history or not detector_results:
        return {}

    total_attacks = 0
    detected_attacks = 0
    total_clients = 0
    false_positives = 0

    for round_data, detection in zip(update_history, detector_results):
        malicious_clients = set(round_data.get("malicious_clients", []))
        detected_clients = set(detection.get("suspicious_clients", []))

        total_attacks += len(malicious_clients)
        detected_attacks += len(malicious_clients & detected_clients)
        total_clients += len(round_data.get("updates", []))
        false_positives += len(detected_clients - malicious_clients)

    # Detection rate
    detection_rate = detected_attacks / total_attacks if total_attacks > 0 else 0.0

    # False positive rate
    false_positive_rate = false_positives / total_clients if total_clients > 0 else 0.0

    return {
        "total_attacks": total_attacks,
        "detected_attacks": detected_attacks,
        "detection_rate": detection_rate,
        "false_positives": false_positives,
        "false_positive_rate": false_positive_rate
    }
