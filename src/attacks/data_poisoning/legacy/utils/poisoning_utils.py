"""
Utility functions for label manipulation and poisoning.

This module provides helper functions for implementing data poisoning attacks,
particularly label flipping attacks in federated learning.
"""

import numpy as np


def flip_labels(labels: np.ndarray, flip_prob: float) -> np.ndarray:
    """
    Randomly flip labels with a given probability.

    Args:
        labels: Original labels (binary: 0 or 1)
        flip_prob: Probability of flipping each label

    Returns:
        Flipped labels as numpy array
    """
    flipped = labels.copy()
    mask = np.random.random(len(labels)) < flip_prob
    flipped[mask] = 1 - flipped[mask]
    return flipped


def flip_fraud_to_legitimate(labels: np.ndarray, flip_prob: float) -> np.ndarray:
    """
    Flip fraud labels (1) to legitimate (0) with given probability.

    This is a targeted attack where only fraud cases are flipped to legitimate.

    Args:
        labels: Original labels (binary: 0 or 1)
        flip_prob: Probability of flipping each fraud label to legitimate

    Returns:
        Flipped labels as numpy array
    """
    flipped = labels.copy()
    fraud_indices = np.where(labels == 1)[0]
    flip_mask = np.random.random(len(fraud_indices)) < flip_prob
    flip_indices = fraud_indices[flip_mask]
    flipped[flip_indices] = 0
    return flipped


def invert_labels(labels: np.ndarray) -> np.ndarray:
    """
    Invert all labels (0 -> 1, 1 -> 0).

    Args:
        labels: Original labels (binary: 0 or 1)

    Returns:
        Inverted labels as numpy array
    """
    return 1 - labels


def select_malicious_clients(
    total_clients: int,
    malicious_indices: list[int] | None = None,
    malicious_fraction: float = 0.2,
    seed: int = 42
) -> list[int]:
    """
    Select which clients are malicious for the attack.

    Args:
        total_clients: Total number of clients in the federation
        malicious_indices: Specific client indices to be malicious (None for random selection)
        malicious_fraction: Fraction of clients that are malicious (if indices not specified)
        seed: Random seed for reproducibility

    Returns:
        List of client indices that are malicious
    """
    if malicious_indices is not None:
        # Validate provided indices
        if not all(0 <= idx < total_clients for idx in malicious_indices):
            raise ValueError(f"Malicious client indices must be between 0 and {total_clients-1}")
        return malicious_indices

    # Randomly select clients
    num_malicious = max(1, int(total_clients * malicious_fraction))
    rng = np.random.default_rng(seed)
    return rng.choice(total_clients, size=num_malicious, replace=False).tolist()


def calculate_flip_statistics(original_labels: np.ndarray, poisoned_labels: np.ndarray) -> dict:
    """
    Calculate statistics about the label flipping.

    Args:
        original_labels: Original labels before attack
        poisoned_labels: Labels after attack

    Returns:
        Dictionary with flip statistics:
            - total_flips: Total number of labels flipped
            - flip_rate: Actual flip rate
            - fraud_to_legitimate: Number of 1 -> 0 flips
            - legitimate_to_fraud: Number of 0 -> 1 flips
    """
    flips = original_labels != poisoned_labels
    total_flips = np.sum(flips)

    fraud_to_leg = np.sum((original_labels == 1) & (poisoned_labels == 0))
    legit_to_fraud = np.sum((original_labels == 0) & (poisoned_labels == 1))

    return {
        "total_flips": int(total_flips),
        "flip_rate": float(total_flips / len(original_labels)),
        "fraud_to_legitimate": int(fraud_to_leg),
        "legitimate_to_fraud": int(legit_to_fraud),
    }
