"""
Metrics for evaluating attack impact in Federated Learning.

This module provides functions to calculate various metrics that quantify
the impact of label flipping attacks on model performance.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import defaultdict
from torch.utils.data import DataLoader


def calculate_attack_success_rate(
    baseline_acc: float,
    attacked_acc: float,
    target_degradation: float = 0.1
) -> Dict[str, float]:
    """
    Calculate attack success rate based on accuracy degradation.

    Args:
        baseline_acc: Accuracy of model without attack
        attacked_acc: Accuracy of model under attack
        target_degradation: Target accuracy degradation (default: 10%)

    Returns:
        Dictionary with attack success metrics:
            - accuracy_degradation: Drop in accuracy (baseline - attacked)
            - relative_degradation: Degradation as percentage of baseline
            - attack_success: Boolean indicating if target degradation achieved
            - success_rate: Ratio of achieved to target degradation
    """
    accuracy_degradation = baseline_acc - attacked_acc
    relative_degradation = (accuracy_degradation / baseline_acc) if baseline_acc > 0 else 0.0
    attack_success = accuracy_degradation >= target_degradation
    success_rate = (accuracy_degradation / target_degradation) if target_degradation > 0 else 0.0

    return {
        "accuracy_degradation": accuracy_degradation,
        "relative_degradation": relative_degradation,
        "attack_success": attack_success,
        "success_rate": success_rate,
    }


def calculate_per_class_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Calculate per-class accuracy (fraud vs legitimate).

    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Dictionary with per-class accuracies:
            - accuracy: Overall accuracy
            - accuracy_legitimate: Accuracy on class 0 (legitimate)
            - accuracy_fraud: Accuracy on class 1 (fraud)
    """
    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            predictions = torch.argmax(logits, dim=1)

            # Overall accuracy
            total_correct += (predictions == y).sum().item()
            total_samples += y.size(0)

            # Per-class accuracy
            for pred, true in zip(predictions.cpu().numpy(), y.cpu().numpy()):
                class_total[true] += 1
                if pred == true:
                    class_correct[true] += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
    }

    # Add per-class accuracy if both classes present
    if 0 in class_total:
        metrics["accuracy_legitimate"] = class_correct[0] / class_total[0]

    if 1 in class_total:
        metrics["accuracy_fraud"] = class_correct[1] / class_total[1]

    return metrics


def calculate_convergence_delay(
    baseline_history: Dict[str, List],
    attacked_history: Dict[str, List],
    threshold: float = 0.01,
    accuracy_key: str = "global_accuracy"
) -> Dict[str, any]:
    """
    Calculate convergence delay caused by the attack.

    Convergence is defined as reaching within 1% of final accuracy.

    Args:
        baseline_history: Training history without attack
        attacked_history: Training history under attack
        threshold: Convergence threshold (relative to final accuracy)
        accuracy_key: Key for accuracy in history dictionaries

    Returns:
        Dictionary with convergence metrics:
            - baseline_convergence_round: Round when baseline converged
            - attacked_convergence_round: Round when attacked model converged
            - convergence_delay: Additional rounds needed to converge
            - baseline_final_accuracy: Final accuracy of baseline
            - attacked_final_accuracy: Final accuracy under attack
    """
    def find_convergence_round(accuracy_list: List[float]) -> int:
        """Find the round where accuracy converges."""
        if not accuracy_list:
            return None

        final_acc = accuracy_list[-1]
        convergence_threshold = max(threshold, final_acc * threshold)

        for i, acc in enumerate(accuracy_list):
            if abs(acc - final_acc) < convergence_threshold:
                return i + 1  # Round numbers are 1-indexed

        return None

    baseline_accs = baseline_history.get(accuracy_key, [])
    attacked_accs = attacked_history.get(accuracy_key, [])

    baseline_round = find_convergence_round(baseline_accs)
    attacked_round = find_convergence_round(attacked_accs)

    convergence_delay = None
    if baseline_round is not None and attacked_round is not None:
        convergence_delay = attacked_round - baseline_round

    return {
        "baseline_convergence_round": baseline_round,
        "attacked_convergence_round": attacked_round,
        "convergence_delay": convergence_delay,
        "baseline_final_accuracy": baseline_accs[-1] if baseline_accs else None,
        "attacked_final_accuracy": attacked_accs[-1] if attacked_accs else None,
    }


def calculate_training_stability(
    history: Dict[str, List],
    window_size: int = 5,
    accuracy_key: str = "global_accuracy"
) -> Dict[str, float]:
    """
    Calculate training stability metrics.

    Args:
        history: Training history
        window_size: Window size for rolling statistics
        accuracy_key: Key for accuracy in history dictionary

    Returns:
        Dictionary with stability metrics:
            - accuracy_variance: Variance in accuracy
            - accuracy_std: Standard deviation of accuracy
            - max_accuracy_drop: Maximum drop in accuracy between consecutive rounds
            - final_accuracy: Final accuracy value
    """
    accuracies = history.get(accuracy_key, [])

    if not accuracies:
        return {
            "accuracy_variance": None,
            "accuracy_std": None,
            "max_accuracy_drop": None,
            "final_accuracy": None,
        }

    acc_array = np.array(accuracies)

    # Calculate variance and std
    variance = np.var(acc_array)
    std = np.std(acc_array)

    # Calculate maximum consecutive drop
    max_drop = 0.0
    for i in range(1, len(acc_array)):
        drop = acc_array[i-1] - acc_array[i]
        if drop > max_drop:
            max_drop = drop

    return {
        "accuracy_variance": variance,
        "accuracy_std": std,
        "max_accuracy_drop": max_drop,
        "final_accuracy": acc_array[-1],
    }


def compare_histories(
    baseline_history: Dict[str, List],
    attacked_history: Dict[str, List],
    accuracy_key: str = "global_accuracy"
) -> Dict[str, any]:
    """
    Compare baseline and attacked training histories.

    Args:
        baseline_history: Training history without attack
        attacked_history: Training history under attack
        accuracy_key: Key for accuracy in history dictionaries

    Returns:
        Dictionary with comparison metrics:
            - final_accuracy_baseline: Final accuracy without attack
            - final_accuracy_attacked: Final accuracy under attack
            - accuracy_drop: Difference in final accuracy
            - attack_success_metrics: Attack success rate metrics
            - convergence_metrics: Convergence delay metrics
            - stability_metrics: Stability comparison
    """
    baseline_accs = baseline_history.get(accuracy_key, [])
    attacked_accs = attacked_history.get(accuracy_key, [])

    final_baseline = baseline_accs[-1] if baseline_accs else None
    final_attacked = attacked_accs[-1] if attacked_accs else None

    accuracy_drop = None
    if final_baseline is not None and final_attacked is not None:
        accuracy_drop = final_baseline - final_attacked

    return {
        "final_accuracy_baseline": final_baseline,
        "final_accuracy_attacked": final_attacked,
        "accuracy_drop": accuracy_drop,
        "attack_success_metrics": calculate_attack_success_rate(
            final_baseline or 0.0,
            final_attacked or 0.0
        ) if final_baseline is not None else None,
        "convergence_metrics": calculate_convergence_delay(
            baseline_history,
            attacked_history,
            accuracy_key=accuracy_key
        ),
        "stability_baseline": calculate_training_stability(
            baseline_history,
            accuracy_key=accuracy_key
        ),
        "stability_attacked": calculate_training_stability(
            attacked_history,
            accuracy_key=accuracy_key
        ),
    }


def calculate_robustness_metrics(
    results: Dict[float, Dict[str, List]],
    baseline_accuracy: float,
    accuracy_key: str = "global_accuracy"
) -> Dict[str, any]:
    """
    Calculate robustness metrics across different attacker fractions.

    Args:
        results: Dictionary mapping attacker fraction to history
        baseline_accuracy: Accuracy without any attack
        accuracy_key: Key for accuracy in history dictionaries

    Returns:
        Dictionary with robustness metrics:
            - degradation_by_fraction: Accuracy degradation per fraction
            - critical_fraction: Smallest fraction causing >5% degradation
            - robustness_score: Area under accuracy curve normalized
    """
    degradation = {}

    for fraction, history in results.items():
        accuracies = history.get(accuracy_key, [])
        final_acc = accuracies[-1] if accuracies else 0.0
        degradation[fraction] = baseline_accuracy - final_acc

    # Find critical fraction (causes >5% degradation)
    critical_fraction = None
    for frac, deg in sorted(degradation.items()):
        if deg > 0.05 * baseline_accuracy:
            critical_fraction = frac
            break

    # Calculate robustness score (normalized area under accuracy curve)
    fractions = sorted(degradation.keys())
    if fractions:
        # Higher is better (less degradation)
        robustness_score = np.mean([baseline_accuracy - d for d in degradation.values()])
    else:
        robustness_score = None

    return {
        "degradation_by_fraction": degradation,
        "critical_fraction": critical_fraction,
        "robustness_score": robustness_score,
        "all_degradations": degradation,
    }
