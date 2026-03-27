"""
Attack scaling strategy for backdoor attacks.
Implements magnitude scaling to help backdoor survive FedAvg averaging.
"""

import torch
from typing import Dict


def compute_scale_factor(
    num_clients: int,
    num_malicious: int
) -> float:
    """
    Compute scale factor for malicious updates.

    To survive FedAvg: malicious_update * scale_factor * (num_malicious / num_clients)
    should equal malicious_update.

    Therefore: scale_factor = num_clients / num_malicious

    Args:
        num_clients: Total number of clients
        num_malicious: Number of malicious clients

    Returns:
        Scale factor
    """
    if num_malicious == 0:
        return 1.0

    return num_clients / num_malicious


def scale_malicious_updates(
    updates: Dict[str, torch.Tensor],
    scale_factor: float
) -> Dict[str, torch.Tensor]:
    """
    Scale malicious updates to survive FedAvg.

    Args:
        updates: Dictionary of parameter updates
        scale_factor: Scaling factor

    Returns:
        Scaled updates
    """
    scaled_updates = {}

    for name, update in updates.items():
        scaled_updates[name] = update * scale_factor

    return scaled_updates


def compute_malicious_direction(
    honest_updates: Dict[str, torch.Tensor],
    backdoor_updates: Dict[str, torch.Tensor],
    alpha: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Combine honest and malicious updates.

    Strategy: Scale malicious updates and add to honest updates.
    This makes the attack harder to detect.

    Args:
        honest_updates: Updates from clean data
        backdoor_updates: Updates from poisoned data
        alpha: Weight for backdoor updates (default=1.0)

    Returns:
        Combined malicious updates
    """
    combined = {}

    for name in honest_updates:
        if name in backdoor_updates:
            # honest_update + alpha * backdoor_update
            combined[name] = honest_updates[name] + alpha * backdoor_updates[name]
        else:
            combined[name] = honest_updates[name]

    return combined


def normalize_updates(
    updates: Dict[str, torch.Tensor],
    max_norm: float = 10.0
) -> Dict[str, torch.Tensor]:
    """
    Normalize updates to prevent extremely large values.

    Args:
        updates: Dictionary of parameter updates
        max_norm: Maximum L2 norm for updates

    Returns:
        Normalized updates
    """
    # Compute total norm
    total_norm = torch.sqrt(
        sum(torch.sum(update ** 2) for update in updates.values())
    )

    if total_norm > max_norm:
        scale = max_norm / total_norm
        for name in updates:
            updates[name] = updates[name] * scale

    return updates


if __name__ == "__main__":
    # Test scaling
    num_clients = 20
    num_malicious = 1

    scale_factor = compute_scale_factor(num_clients, num_malicious)
    print(f"Scale factor: {scale_factor}")

    # Test on dummy updates
    updates = {
        'layer1.weight': torch.randn(64, 30),
        'layer1.bias': torch.randn(64),
        'layer2.weight': torch.randn(32, 64),
    }

    scaled = scale_malicious_updates(updates, scale_factor)

    print(f"Original mean magnitude: {torch.mean(torch.abs(updates['layer1.weight']))}")
    print(f"Scaled mean magnitude: {torch.mean(torch.abs(scaled['layer1.weight']))}")
