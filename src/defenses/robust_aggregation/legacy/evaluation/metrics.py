"""
Evaluation metrics for measuring aggregator robustness.

This module provides metrics to evaluate both model accuracy and
attack success rate for robust aggregation methods.
"""

from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> float:
    """
    Compute model accuracy on test dataset.

    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader with test data
        device: Device to run evaluation on ('cpu' or 'cuda')

    Returns:
        Accuracy as a float between 0 and 1
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_attack_success_rate(
    model: nn.Module,
    test_loader: DataLoader,
    attack_type: str,
    device: str = 'cpu',
    target_label: int = None,
    backdoor_pattern: torch.Tensor = None
) -> float:
    """
    Compute attack success rate (ASR) for a given attack type.

    For label flipping: measures how often flipped labels are predicted
    For backdoor: measures how often backdoor triggers misclassification
    For model poisoning: measures degradation in accuracy or targeted misclassification

    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader with adversarial test data
        attack_type: Type of attack ('label_flipping', 'backdoor', 'model_poisoning')
        device: Device to run evaluation on
        target_label: For backdoor attacks, the target label to trigger
        backdoor_pattern: For backdoor attacks, the pattern to inject

    Returns:
        Attack success rate as a float between 0 and 1

    Note:
        Higher ASR is worse (means attack succeeded more)
        Lower ASR is better (means aggregator defended against attack)
    """
    model.eval()
    model.to(device)

    if attack_type == 'label_flipping':
        # ASR: How often model predicts flipped (incorrect) label
        successful_attacks = 0
        total = 0

        with torch.no_grad():
            for inputs, true_targets in test_loader:
                inputs = inputs.to(device)
                true_targets = true_targets.to(device)

                # For label flipping, test_loader contains flipped labels
                # Model should predict original labels, not flipped ones
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)

                # Count when model predicts flipped label (attack success)
                # This assumes test_loader provides (flipped_input, original_label)
                # and we want model to predict original_label
                successful_attacks += (predicted != true_targets).sum().item()
                total += true_targets.size(0)

        asr = successful_attacks / total if total > 0 else 0.0
        return asr

    elif attack_type == 'backdoor':
        # ASR: How often backdoor pattern causes target prediction
        if target_label is None:
            raise ValueError("target_label required for backdoor ASR computation")

        successful_attacks = 0
        total = 0

        with torch.no_grad():
            for inputs, _ in test_loader:
                # Inject backdoor pattern if provided
                if backdoor_pattern is not None:
                    inputs = inputs.clone()
                    # Simple pattern injection (e.g., set bottom-right corner)
                    pattern_size = int(backdoor_pattern.numel() ** 0.5)
                    inputs[:, :, -pattern_size:, -pattern_size:] = backdoor_pattern

                inputs = inputs.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)

                # Count predictions of target label (attack success)
                successful_attacks += (predicted == target_label).sum().item()
                total += inputs.size(0)

        asr = successful_attacks / total if total > 0 else 0.0
        return asr

    elif attack_type == 'model_poisoning':
        # ASR for model poisoning: measure targeted degradation
        # This is attack-specific, so we measure accuracy drop
        # Higher accuracy drop means more successful attack
        baseline_acc = compute_accuracy(model, test_loader, device)

        # For model poisoning, ASR = 1 - accuracy (more degradation = higher ASR)
        asr = 1.0 - baseline_acc
        return asr

    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def compute_convergence_speed(
    accuracy_history: List[float],
    target_threshold: float = 0.90,
    min_accuracy: float = 0.0
) -> int:
    """
    Compute number of rounds to reach target accuracy threshold.

    Convergence speed is measured as the number of training rounds
    required for the model to first reach or exceed the target accuracy.

    Args:
        accuracy_history: List of accuracy values per training round
        target_threshold: Target accuracy threshold (default: 0.90)
        min_accuracy: Minimum starting accuracy to consider (default: 0.0)

    Returns:
        Number of rounds to reach threshold, or -1 if never reached

    Example:
        >>> history = [0.5, 0.7, 0.85, 0.92, 0.95]
        >>> compute_convergence_speed(history, threshold=0.90)
        3  # Reached 0.92 at round 3 (0-indexed)

    Note:
        Faster convergence (lower value) is better
        If threshold is never reached, returns len(accuracy_history)
    """
    for round_idx, accuracy in enumerate(accuracy_history):
        if accuracy >= target_threshold and accuracy >= min_accuracy:
            return round_idx

    # Never reached threshold
    return len(accuracy_history)


def compute_defense_effectiveness(
    clean_accuracy: float,
    attacked_accuracy: float,
    clean_asr: float = 0.0,
    attacked_asr: float = 0.0
) -> Dict[str, float]:
    """
    Compute overall defense effectiveness metrics.

    Args:
        clean_accuracy: Model accuracy without attacks
        attacked_accuracy: Model accuracy with attacks (after defense)
        clean_asr: Attack success rate without defense (should be high)
        attacked_asr: Attack success rate with defense (should be low)

    Returns:
        Dict with effectiveness metrics:
        - 'accuracy_preservation': How much accuracy is preserved (0-1, higher is better)
        - 'asr_reduction': How much ASR is reduced (0-1, higher is better)
        - 'defense_score': Combined effectiveness score (0-1, higher is better)
    """
    # Accuracy preservation: ratio of attacked to clean accuracy
    accuracy_preservation = attacked_accuracy / clean_accuracy if clean_accuracy > 0 else 0.0

    # ASR reduction: proportional reduction in attack success
    if clean_asr > 0:
        asr_reduction = (clean_asr - attacked_asr) / clean_asr
    else:
        # If no baseline attack, measure ASR directly (lower is better)
        asr_reduction = 1.0 - attacked_asr

    # Combined defense score: average of both metrics
    defense_score = (accuracy_preservation + asr_reduction) / 2.0

    return {
        'accuracy_preservation': accuracy_preservation,
        'asr_reduction': asr_reduction,
        'defense_score': defense_score,
    }


def compute_aggregator_statistics(
    results: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics across multiple experimental runs.

    Args:
        results: List of result dicts, each containing metrics like
                'accuracy', 'asr', 'convergence_speed'

    Returns:
        Dict with 'mean', 'std', 'min', 'max' for each metric
    """
    if not results:
        return {}

    metrics = {}
    metric_names = results[0].keys()

    for metric_name in metric_names:
        values = [r[metric_name] for r in results if metric_name in r]

        if values:
            metrics[metric_name] = {
                'mean': sum(values) / len(values),
                'std': (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5,
                'min': min(values),
                'max': max(values),
                'count': len(values),
            }

    return metrics
