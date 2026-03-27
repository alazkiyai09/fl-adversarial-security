"""
Robust aggregation methods for comparison with FoolsGold.

Implements:
- Krum
- Multi-Krum
- Trimmed Mean
"""

import numpy as np
from typing import List, Tuple
from flwr.common import Parameters, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from .base import BaseAggregator
from ..utils.similarity import flatten_parameters


def compute_euclidean_distances(
    gradients: List[np.ndarray]
) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix.

    Args:
        gradients: List of gradient vectors

    Returns:
        Distance matrix D where D[i,j] = ||grad_i - grad_j||
    """
    n = len(gradients)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist = np.linalg.norm(gradients[i] - gradients[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def krum_select(
    gradients: List[np.ndarray],
    num_malicious: int = 1
) -> int:
    """
    Krum: Select client with smallest sum of distances to nearest neighbors.

    Args:
        gradients: List of gradient vectors
        num_malicious: Expected number of malicious clients (f)

    Returns:
        Index of selected client
    """
    n = len(gradients)
    if n == 0:
        raise ValueError("No gradients to select from")

    # Number of neighbors to consider: n - f - 2
    num_neighbors = n - num_malicious - 2
    num_neighbors = max(1, num_neighbors)

    # Compute distances
    distances = compute_euclidean_distances(gradients)

    # For each client, compute sum of distances to nearest neighbors
    scores = []
    for i in range(n):
        # Get distances to all other clients
        row_distances = [(j, distances[i, j]) for j in range(n) if j != i]
        # Sort by distance
        row_distances.sort(key=lambda x: x[1])
        # Sum to num_neighbors closest
        score = sum(d for _, d in row_distances[:num_neighbors])
        scores.append(score)

    # Select client with minimum score
    return int(np.argmin(scores))


def multi_krum_select(
    gradients: List[np.ndarray],
    num_malicious: int = 1,
    num_selected: int = None
) -> List[int]:
    """
    Multi-Krum: Select multiple clients using Krum iteratively.

    Args:
        gradients: List of gradient vectors
        num_malicious: Expected number of malicious clients
        num_selected: Number of clients to select (default: n - 2f)

    Returns:
        List of selected client indices
    """
    n = len(gradients)
    if n == 0:
        return []

    if num_selected is None:
        num_selected = n - 2 * num_malicious
    num_selected = max(1, min(num_selected, n))

    selected = []
    remaining = list(range(n))
    remaining_grads = gradients.copy()

    for _ in range(num_selected):
        if not remaining:
            break
        # Select using Krum
        idx = krum_select(remaining_grads, num_malicious)
        selected.append(remaining[idx])
        # Remove selected
        remaining.pop(idx)
        remaining_grads.pop(idx)

    return selected


def trimmed_mean_aggregate(
    parameters: List[Parameters],
    trim_ratio: float = 0.1
) -> Parameters:
    """
    Trimmed Mean: Remove extreme values and average the rest.

    For each parameter dimension, remove highest and lowest trim_ratio
    fraction, then average the remaining.

    Args:
        parameters: List of client parameters
        trim_ratio: Fraction to trim from each end (0-0.5)

    Returns:
        Aggregated parameters
    """
    num_clients = len(parameters)
    if num_clients == 0:
        raise ValueError("Cannot aggregate empty parameter list")

    # Convert to numpy arrays
    all_ndarrays = [parameters_to_ndarrays(p) for p in parameters]
    num_layers = len(all_ndarrays[0])

    aggregated_ndarrays = []
    for layer_idx in range(num_layers):
        # Stack all client parameters for this layer
        layer_params = np.stack([all_ndarrays[c][layer_idx] for c in range(num_clients)])

        # Number to trim from each end
        num_trim = max(1, int(num_clients * trim_ratio))

        # Flatten for sorting
        original_shape = layer_params.shape[1:]
        flat_params = layer_params.reshape(num_clients, -1)

        # Trim and average for each dimension
        aggregated_flat = np.zeros(flat_params.shape[1])
        for dim in range(flat_params.shape[1]):
            values = flat_params[:, dim]
            # Sort
            sorted_idx = np.argsort(values)
            # Remove extremes
            trimmed_idx = sorted_idx[num_trim:num_clients - num_trim]
            # Average remaining
            aggregated_flat[dim] = np.mean(values[trimmed_idx])

        # Reshape back
        aggregated_layer = aggregated_flat.reshape(original_shape)
        aggregated_ndarrays.append(aggregated_layer)

    return ndarrays_to_parameters(aggregated_ndarrays)


class KrumAggregator(BaseAggregator):
    """Krum robust aggregator."""

    def __init__(self, num_malicious: int = 1):
        super().__init__(num_malicious=num_malicious)
        self.num_malicious = num_malicious

    def aggregate(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Parameters:
        if not results:
            raise ValueError("No results to aggregate")

        # Flatten gradients
        gradients = [flatten_parameters(r.parameters) for _, r in results]

        # Select using Krum
        selected_idx = krum_select(gradients, self.num_malicious)

        # Track
        self.history["selected_indices"] = self.history.get("selected_indices", []) + [selected_idx]

        # Return selected parameters
        return results[selected_idx][1].parameters


class MultiKrumAggregator(BaseAggregator):
    """Multi-Krum robust aggregator."""

    def __init__(self, num_malicious: int = 1, num_selected: int = None):
        super().__init__(num_malicious=num_malicious, num_selected=num_selected)
        self.num_malicious = num_malicious
        self.num_selected = num_selected

    def aggregate(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Parameters:
        if not results:
            raise ValueError("No results to aggregate")

        # Flatten gradients
        gradients = [flatten_parameters(r.parameters) for _, r in results]

        # Select using Multi-Krum
        selected_indices = multi_krum_select(
            gradients,
            self.num_malicious,
            self.num_selected
        )

        # Track
        self.history["selected_indices"] = self.history.get("selected_indices", []) + [selected_indices]

        # Average selected parameters
        selected_params = [results[i][1].parameters for i in selected_indices]
        return trimmed_mean_aggregate(selected_params, trim_ratio=0.0)


class TrimmedMeanAggregator(BaseAggregator):
    """Trimmed Mean robust aggregator."""

    def __init__(self, trim_ratio: float = 0.1):
        super().__init__(trim_ratio=trim_ratio)
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Parameters:
        if not results:
            raise ValueError("No results to aggregate")

        # Extract parameters
        parameters = [r.parameters for _, r in results]

        # Aggregate with trimmed mean
        return trimmed_mean_aggregate(parameters, self.trim_ratio)
