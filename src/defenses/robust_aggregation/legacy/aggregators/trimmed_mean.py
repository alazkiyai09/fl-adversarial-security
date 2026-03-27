"""
Trimmed Mean Aggregation.

Reference: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
(Chen et al., NeurIPS 2017)

Mathematical definition:
    1. Sort each parameter's values across clients
    2. Remove β smallest and β largest values (where β = alpha * n)
    3. Average the remaining values

Robustness guarantee: Can tolerate up to β Byzantine clients.
"""

from typing import Dict, List
import torch

from .base import RobustAggregator


class TrimmedMean(RobustAggregator):
    """
    Trimmed mean aggregation for Byzantine-resilience.

    For each parameter independently:
    1. Sort values across all clients
    2. Remove the top and bottom β values (β = alpha × n)
    3. Average the remaining (n - 2β) values

    This is robust because it explicitly removes extreme values that could
    be from Byzantine clients.

    Robustness: Tolerates up to β = alpha × n Byzantine clients
    Complexity: O(P × n log n) per parameter where P = number of parameters

    Args:
        beta: Fraction of clients to trim from each end (default: 0.2)
              Must satisfy: 0 < beta < 0.5
              Total trimmed = 2β × n clients

    Example:
        >>> aggregator = TrimmedMean(beta=0.2)
        >>> updates = [{'layer.weight': tensor(...)}, ...]  # n=10 clients
        >>> # Removes lowest 2 and highest 2, averages middle 6
        >>> aggregated = aggregator.aggregate(updates, num_attackers=2)
    """

    def __init__(self, beta: float = 0.2):
        """
        Initialize trimmed mean aggregator.

        Args:
            beta: Fraction of clients to trim from EACH end
                  Total trimmed = 2 × beta × n clients

        Raises:
            ValueError: If beta is not in valid range (0, 0.5)
        """
        if not (0 < beta < 0.5):
            raise ValueError(
                f"beta must be in (0, 0.5), got {beta}. "
                f"beta=0.5 would trim all clients, beta=0 is just mean."
            )

        self.beta = beta

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        num_attackers: int
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using trimmed mean.

        Args:
            updates: List of n model updates from clients
            num_attackers: Maximum number of Byzantine clients

        Returns:
            Aggregated model update

        Raises:
            ValueError: If updates is empty, insufficient clients, or mismatched structure
        """
        self._validate_updates(updates)

        n = len(updates)
        k = int(self.beta * n)  # Number to trim from each end

        # Verify sufficient clients for robustness
        # Need n > 2k to have remaining clients after trimming
        if n <= 2 * k:
            raise ValueError(
                f"Trimmed mean requires n > 2βn after trimming. "
                f"Got n={n}, k={k} (beta={self.beta}), would have {n - 2*k} clients left"
            )

        # Verify we can handle the expected number of attackers
        if num_attackers > k:
            raise ValueError(
                f"Trimmed mean with beta={self.beta} can only handle k={k} attackers, "
                f"but num_attackers={num_attackers}"
            )

        aggregated = {}

        # Compute trimmed mean for each parameter independently
        for param_name in updates[0].keys():
            # Stack all client values for this parameter
            stacked = torch.stack([u[param_name] for u in updates])

            # Sort along the client dimension (dim=0)
            # For multi-dimensional parameters, sort along first axis only
            sorted_values, _ = torch.sort(stacked, dim=0)

            # Trim: remove k smallest and k largest
            # Keep indices [k, n-k)
            trimmed = sorted_values[k:n - k]

            # Average the remaining values
            mean_value = torch.mean(trimmed, dim=0)

            aggregated[param_name] = mean_value

        return aggregated

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(beta={self.beta})"


def trimmed_mean(
    updates: List[Dict[str, torch.Tensor]],
    num_attackers: int,
    beta: float = 0.2
) -> Dict[str, torch.Tensor]:
    """
    Functional interface for trimmed mean aggregation.

    Equivalent to TrimmedMean(beta=beta).aggregate(updates, num_attackers)

    Args:
        updates: List of model updates from clients
        num_attackers: Maximum number of Byzantine clients
        beta: Fraction of clients to trim from each end

    Returns:
        Aggregated model update
    """
    aggregator = TrimmedMean(beta=beta)
    return aggregator.aggregate(updates, num_attackers)
