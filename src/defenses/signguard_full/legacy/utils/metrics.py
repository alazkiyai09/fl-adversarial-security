"""Evaluation metrics for SignGuard experiments."""

import torch
import time
from typing import Dict, List
from torch.utils.data import DataLoader


def compute_accuracy(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Compute model accuracy on test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Accuracy as float [0, 1]
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total if total > 0 else 0.0


def compute_attack_success_rate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    target_class: int,
    trigger_pattern: torch.Tensor | None = None,
    device: str = "cpu",
) -> float:
    """Compute attack success rate.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        target_class: Target class for attack
        trigger_pattern: Optional backdoor trigger pattern
        device: Device to run on

    Returns:
        Attack success rate as float [0, 1]
    """
    model.eval()
    successful = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Apply trigger if backdoor attack
            if trigger_pattern is not None:
                inputs = apply_trigger(inputs, trigger_pattern)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # For backdoor: check if trigger samples are classified as target
            if trigger_pattern is not None:
                successful += (predicted == target_class).sum().item()
                total += targets.size(0)
            # For label flip: check if source class is classified as target
            else:
                source_mask = targets != target_class
                if source_mask.any():
                    successful += (
                        (predicted[source_mask] == target_class).sum().item()
                    )
                    total += source_mask.sum().item()

    return successful / total if total > 0 else 0.0


def compute_detection_metrics(
    true_malicious: List[str],
    detected_malicious: List[str],
    all_clients: List[str],
) -> tuple[float, float, float]:
    """Compute detection F1, precision, recall.

    Args:
        true_malicious: List of actual malicious client IDs
        detected_malicious: List of detected malicious client IDs
        all_clients: List of all client IDs

    Returns:
        (precision, recall, f1) tuple
    """
    true_set = set(true_malicious)
    detected_set = set(detected_malicious)

    true_positives = len(true_set & detected_set)
    false_positives = len(detected_set - true_set)
    false_negatives = len(true_set - detected_set)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def compute_overhead_metrics(
    execution_times: Dict[str, List[float]],
    communication_bytes: Dict[str, List[int]],
    memory_usage: Dict[str, List[int]],
) -> Dict[str, float]:
    """Compute time, communication, and memory overhead.

    Args:
        execution_times: Dict mapping client_id -> list of execution times
        communication_bytes: Dict mapping client_id -> list of bytes sent
        memory_usage: Dict mapping client_id -> list of memory usage

    Returns:
        Dictionary with aggregated overhead metrics
    """
    # Average execution time
    all_times = [t for times in execution_times.values() for t in times]
    avg_time = sum(all_times) / len(all_times) if all_times else 0.0

    # Total communication
    all_bytes = [b for bytes_list in communication_bytes.values() for b in bytes_list]
    total_bytes = sum(all_bytes) if all_bytes else 0

    # Average memory usage
    all_memory = [m for mem_list in memory_usage.values() for m in mem_list]
    avg_memory = sum(all_memory) / len(all_memory) if all_memory else 0.0

    return {
        "avg_execution_time": avg_time,
        "total_communication_bytes": total_bytes,
        "avg_memory_mb": avg_memory / (1024 * 1024),  # Convert to MB
    }


def apply_trigger(inputs: torch.Tensor, trigger_pattern: torch.Tensor) -> torch.Tensor:
    """Apply backdoor trigger to inputs.

    Args:
        inputs: Input tensor
        trigger_pattern: Trigger pattern to apply

    Returns:
        Modified inputs with trigger
    """
    inputs = inputs.clone()
    batch_size = inputs.size(0)
    trigger_size = min(trigger_pattern.numel(), inputs.numel() // batch_size)

    for i in range(batch_size):
        inputs[i].view(-1)[:trigger_size] = trigger_pattern[:trigger_size]

    return inputs
