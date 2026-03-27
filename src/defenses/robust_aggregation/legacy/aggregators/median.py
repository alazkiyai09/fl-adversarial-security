"""
Coordinate-wise Median Aggregation.

Reference: "Distributed Optimization with Arbitrary Adversaries" (Chen et al., 2017)

Mathematical definition:
    For each parameter θ, the aggregated value is median(θ₁, θ₂, ..., θₙ)

Robustness guarantee: Can tolerate up to floor(n/2) Byzantine clients.
"""

from typing import Dict, List
import torch
import torch.nn as nn

from .base import RobustAggregator


class CoordinateWiseMedian(RobustAggregator):
    """
    Coordinate-wise median aggregation for Byzantine-resilience.

    For each parameter independently, computes the median across all client updates.
    This is robust because the median is not heavily influenced by extreme values.

    Robustness: Tolerates up to floor(n/2) Byzantine clients where n = total clients
    Complexity: O(P × n log n) per parameter where P = number of parameters

    Example:
        >>> aggregator = CoordinateWiseMedian()
        >>> updates = [{'layer.weight': tensor(...)}, ...]  # n clients
        >>> aggregated = aggregator.aggregate(updates, num_attackers=2)
    """

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        num_attackers: int
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using coordinate-wise median.

        Args:
            updates: List of n model updates from clients
            num_attackers: Maximum number of Byzantine clients (not used directly,
                          but needed for API consistency)

        Returns:
            Aggregated model with each parameter being the median value

        Raises:
            ValueError: If updates is empty or structure mismatched
        """
        self._validate_updates(updates)

        n_clients = len(updates)

        # Verify sufficient clients for robustness
        # Median requires n > 2f to guarantee correctness with f attackers
        if num_attackers > 0 and n_clients <= 2 * num_attackers:
            raise ValueError(
                f"Coordinate-wise median requires n > 2f for robustness. "
                f"Got n={n_clients}, f={num_attackers}"
            )

        aggregated = {}

        # Compute median for each parameter independently
        for param_name in updates[0].keys():
            # Stack all client values for this parameter
            stacked = torch.stack([u[param_name] for u in updates])

            # Compute median along the client dimension (dim=0)
            # torch.median returns (values, indices), we only need values
            median_value, _ = torch.median(stacked, dim=0)

            aggregated[param_name] = median_value

        return aggregated

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def coordinate_wise_median(
    updates: List[Dict[str, torch.Tensor]],
    num_attackers: int
) -> Dict[str, torch.Tensor]:
    """
    Functional interface for coordinate-wise median aggregation.

    Equivalent to CoordinateWiseMedian().aggregate(updates, num_attackers)

    Args:
        updates: List of model updates from clients
        num_attackers: Maximum number of Byzantine clients

    Returns:
        Aggregated model update
    """
    aggregator = CoordinateWiseMedian()
    return aggregator.aggregate(updates, num_attackers)
