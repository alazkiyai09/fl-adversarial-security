"""
Geometric utilities for distance-based robust aggregators.

Implements Euclidean distance computations used by Krum and Bulyan.
"""

from typing import List
import torch


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Euclidean distance between two flattened tensors.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Euclidean distance as a float

    Note:
        Both tensors are flattened to 1D before distance computation.
        This treats all model parameters as a single vector.
    """
    return torch.norm(x.flatten() - y.flatten(), p=2).item()


def pairwise_distances(updates: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between all client updates.

    For n clients, returns an n×n distance matrix where entry [i,j] is
    the distance between updates[i] and updates[j].

    Args:
        updates: List of model updates, where each update is a single flattened
                tensor containing all model parameters

    Returns:
        Distance matrix of shape (n, n) where n = len(updates)
        Diagonal entries are zero (distance from self to self)

    Complexity: O(n² × P) where n = number of clients, P = number of parameters
    """
    n = len(updates)
    distances = torch.zeros(n, n)

    # Compute upper triangular portion (avoid redundant computation)
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(updates[i], updates[j])
            distances[i, j] = dist
            distances[j, i] = dist  # Symmetric

    return distances


def flatten_update(update: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Flatten a model update dict into a single 1D tensor.

    Args:
        update: Dict mapping parameter names to tensors

    Returns:
        Single 1D tensor containing all parameters concatenated
    """
    return torch.cat([param.flatten() for param in update.values()])
