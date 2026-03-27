"""
Krum and Multi-Krum Aggregation.

References:
- "Machine Learning with Adversaries" (Blanchard et al., NeurIPS 2017)
- "Byzantine-Robust Distributed Learning: Towards Optimal Accuracy-Tradeoff"
  (Blanchard et al., NeurIPS 2017)

Krum:
    Selects the update closest to the majority of other updates.
    Minimizes sum of distances to closest (n-f-2) other updates.

Multi-Krum:
    Selects the m updates with smallest Krum scores, then averages them.
    Provides better gradient estimation than single Krum.

Robustness guarantee: Both tolerate up to floor((n-2)/3) Byzantine clients.
"""

from typing import Dict, List
import torch

from .base import RobustAggregator
from ..utils.geometry import pairwise_distances, flatten_update


class Krum(RobustAggregator):
    """
    Krum aggregation for Byzantine-resilience.

    Krum selects the client update that is closest to the majority of other updates.
    Specifically, it chooses the update i that minimizes the sum of distances to
    the closest (n-f-2) other updates.

    Intuition: Malicious updates will likely be far from honest updates, so Krum
    picks the one most "central" to the honest cluster.

    Robustness: Tolerates up to floor((n-2)/3) Byzantine clients
    Complexity: O(n² × P) where n = clients, P = parameters

    Args:
        num_attackers: Number of Byzantine clients f (for API consistency,
                      also auto-computed if not provided)

    Example:
        >>> aggregator = Krum()
        >>> updates = [{'layer.weight': tensor(...)}, ...]  # n=10 clients
        >>> # Selects update closest to other 7 honest updates
        >>> aggregated = aggregator.aggregate(updates, num_attackers=3)
    """

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        num_attackers: int
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using Krum: select the most "central" update.

        Args:
            updates: List of n model updates from clients
            num_attackers: Number of Byzantine clients (f)

        Returns:
            Aggregated model update (single selected update)

        Raises:
            ValueError: If insufficient clients for robustness
        """
        self._validate_updates(updates)

        n = len(updates)

        # Krum requires n >= 2f + 1 for robustness
        # More precisely: n > 3f, so f < floor((n-2)/3)
        max_f = (n - 2) // 3
        if num_attackers > max_f:
            raise ValueError(
                f"Krum requires n >= 3f + 3. Got n={n}, f={num_attackers}, "
                f"maximum f={max_f}"
            )

        # Flatten all updates to vectors for distance computation
        flattened = [flatten_update(u) for u in updates]

        # Compute pairwise distances between all updates
        distances = pairwise_distances(flattened)

        # Compute Krum scores for each update
        krum_scores = self._compute_krum_scores(distances, num_attackers, n)

        # Select the update with minimum Krum score (most central)
        selected_idx = torch.argmin(krum_scores).item()

        return updates[selected_idx].copy()

    def _compute_krum_scores(
        self,
        distances: torch.Tensor,
        num_attackers: int,
        n: int
    ) -> torch.Tensor:
        """
        Compute Krum score for each update.

        The Krum score for update i is the sum of distances to the
        closest (n - f - 2) other updates.

        Args:
            distances: n×n distance matrix
            num_attackers: Number of Byzantine clients f
            n: Total number of clients

        Returns:
            Tensor of Krum scores (length n)
        """
        # Number of closest neighbors to consider
        # This is n - f - 2 as per the paper
        num_closest = n - num_attackers - 2

        if num_closest <= 0:
            raise ValueError(
                f"Cannot compute Krum scores with n={n}, f={num_attackers}. "
                f"Need n - f - 2 > 0, got {num_closest}"
            )

        scores = torch.zeros(n)

        for i in range(n):
            # Get distances from update i to all others
            dists_i = distances[i]

            # Sort distances and take sum of num_closest smallest
            # Exclude self-distance (which is 0)
            sorted_dists, _ = torch.sort(dists_i)
            scores[i] = torch.sum(sorted_dists[1:num_closest + 1])

        return scores

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MultiKrum(RobustAggregator):
    """
    Multi-Krum aggregation for Byzantine-resilience.

    Multi-Krum extends Krum by:
    1. Computing Krum scores for all updates
    2. Selecting the m updates with smallest scores
    3. Averaging those m updates

    This provides better gradient estimation than single Krum by combining
    multiple "good" updates instead of just one.

    Robustness: Same as Krum - tolerates up to floor((n-2)/3) Byzantine clients
    Complexity: O(n² × P) + O(m × P) where m = number of updates to select

    Args:
        m: Number of updates to select and average (default: None, auto-computed)
           If None, m = n - f - 2 (number of non-Byzantine clients expected)

    Example:
        >>> aggregator = MultiKrum(m=5)
        >>> updates = [{'layer.weight': tensor(...)}, ...]  # n=10 clients
        >>> # Selects 5 most central updates and averages them
        >>> aggregated = aggregator.aggregate(updates, num_attackers=3)
    """

    def __init__(self, m: int = None):
        """
        Initialize Multi-Krum aggregator.

        Args:
            m: Number of updates to select and average
               If None, defaults to n - f - 2

        Raises:
            ValueError: If m is not positive
        """
        if m is not None and m <= 0:
            raise ValueError(f"m must be positive, got {m}")

        self.m = m

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        num_attackers: int
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using Multi-Krum: select m central updates and average.

        Args:
            updates: List of n model updates from clients
            num_attackers: Number of Byzantine clients f

        Returns:
            Aggregated model update (average of m selected updates)

        Raises:
            ValueError: If insufficient clients for robustness
        """
        self._validate_updates(updates)

        n = len(updates)

        # Krum requires n >= 2f + 1 for robustness
        max_f = (n - 2) // 3
        if num_attackers > max_f:
            raise ValueError(
                f"Multi-Krum requires n >= 3f + 3. Got n={n}, f={num_attackers}, "
                f"maximum f={max_f}"
            )

        # Determine m if not specified
        m = self.m if self.m is not None else n - num_attackers - 2

        if m > n:
            raise ValueError(
                f"m={m} cannot be larger than n={n}"
            )

        if m <= num_attackers:
            raise ValueError(
                f"m={m} must be greater than num_attackers={num_attackers} "
                f"to exclude Byzantine updates"
            )

        # Flatten all updates to vectors for distance computation
        flattened = [flatten_update(u) for u in updates]

        # Compute pairwise distances between all updates
        distances = pairwise_distances(flattened)

        # Compute Krum scores
        krum = Krum()
        krum_scores = krum._compute_krum_scores(distances, num_attackers, n)

        # Select m updates with smallest Krum scores
        _, top_m_indices = torch.topk(krum_scores, m, largest=False)

        # Average the selected updates
        aggregated = {}
        for param_name in updates[0].keys():
            # Stack selected updates for this parameter
            selected_updates = [updates[i.item()][param_name] for i in top_m_indices]
            stacked = torch.stack(selected_updates)

            # Compute average
            aggregated[param_name] = torch.mean(stacked, dim=0)

        return aggregated

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(m={self.m})"


def krum(
    updates: List[Dict[str, torch.Tensor]],
    num_attackers: int
) -> Dict[str, torch.Tensor]:
    """Functional interface for Krum aggregation."""
    aggregator = Krum()
    return aggregator.aggregate(updates, num_attackers)


def multi_krum(
    updates: List[Dict[str, torch.Tensor]],
    num_attackers: int,
    m: int = None
) -> Dict[str, torch.Tensor]:
    """Functional interface for Multi-Krum aggregation."""
    aggregator = MultiKrum(m=m)
    return aggregator.aggregate(updates, num_attackers)
