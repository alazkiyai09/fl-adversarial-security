"""
Robust aggregation defenses: FedAvg, Median, TrimmedMean, Krum, MultiKrum, Bulyan.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from numpy.typing import NDArray

from .base import BaseDefense


class FedAvgDefense(BaseDefense):
    """
    Federated Averaging (FedAvg) - baseline defense without robustness.

    Simply takes the mean of all client updates. This has no defense against
    Byzantine attacks but serves as a baseline for comparison.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FedAvg defense.

        Args:
            config: Configuration dict (not used for FedAvg)
        """
        super().__init__(config)

    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate updates using simple averaging.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters (mean of all updates)
        """
        params = self._extract_updates(updates)
        return np.mean(params, axis=0)


class MedianDefense(BaseDefense):
    """
    Coordinate-wise median aggregation.

    Takes the median of each parameter dimension across all clients.
    Robust to up to 50% Byzantine clients.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Median defense.

        Args:
            config: Configuration dict (not used for Median)
        """
        super().__init__(config)

    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate updates using coordinate-wise median.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters (median of all updates)
        """
        params = self._extract_updates(updates)
        return np.median(params, axis=0)


class TrimmedMeanDefense(BaseDefense):
    """
    Trimmed Mean aggregation.

    Removes the smallest and largest updates along each dimension before
    averaging. More robust than FedAvg against Byzantine attacks.

    By default, trims the smallest and largest β fraction (e.g., β=0.1 removes
    top and bottom 10%).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Trimmed Mean defense.

        Args:
            config: Configuration with keys:
                - beta: Fraction to trim from each end (default: 0.1)
        """
        super().__init__(config)
        self.beta = config.get("beta", 0.1)

    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate updates using trimmed mean.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters (trimmed mean of all updates)
        """
        params = self._extract_updates(updates)
        n_clients = len(params)

        # Number of clients to trim from each end
        k = int(n_clients * self.beta)

        # Sort along client dimension
        sorted_params = np.sort(params, axis=0)

        # Trim smallest and largest k
        if k > 0:
            trimmed = sorted_params[k:n_clients - k]
        else:
            trimmed = sorted_params

        # Average remaining
        return np.mean(trimmed, axis=0)


class KrumDefense(BaseDefense):
    """
    Krum aggregation algorithm.

    Selects the client update that is most similar to others based on
    Euclidean distance. Robust to Byzantine attacks when the number of
    attackers is less than (n_clients - 2) / 2.

    Reference: Blanchard et al. "Machine learning with adversaries: Byzantine
    tolerant gradient descent." NIPS 2017.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Krum defense.

        Args:
            config: Configuration with keys:
                - num_malicious: Estimated number of malicious clients (default: 0)
                - multi_krum: Whether to use MultiKrum (average of top-k) (default: False)
        """
        super().__init__(config)
        self.num_malicious = config.get("num_malicious", 0)
        self.multi_krum = config.get("multi_krum", False)

        # For MultiKrum
        self.num_selection = config.get("num_selection", None)

    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate updates using Krum or MultiKrum.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters
        """
        params = self._extract_updates(updates)
        n_clients = len(params)

        # Calculate number of closest updates to consider
        f = self.num_malicious if self.num_malicious > 0 else 0
        num_closest = n_clients - f - 2

        # Compute pairwise distances
        scores = self._compute_krum_scores(params)

        if self.multi_krum:
            # MultiKrum: average top-k updates
            k = self.num_selection if self.num_selection else (n_clients - f - 1)
            top_k_indices = np.argsort(scores)[:k]
            return np.mean(params[top_k_indices], axis=0)
        else:
            # Standard Krum: select single best
            best_idx = np.argmin(scores)
            return params[best_idx]

    def _compute_krum_scores(self, params: NDArray) -> NDArray:
        """
        Compute Krum scores for each update.

        Args:
            params: Array of shape (n_clients, n_params)

        Returns:
            Array of scores for each client
        """
        n_clients = len(params)
        f = self.num_malicious if self.num_malicious > 0 else 0
        num_closest = n_clients - f - 2

        # Compute pairwise squared distances
        scores = np.zeros(n_clients)

        for i in range(n_clients):
            distances = np.sum((params - params[i]) ** 2, axis=1)
            # Sum of distances to num_closest nearest neighbors
            scores[i] = np.sum(np.sort(distances)[:num_closest])

        return scores


class MultiKrumDefense(BaseDefense):
    """
    MultiKrum aggregation.

    Averaging of multiple Krum-selected updates. More stable than single Krum
    and provides better empirical performance.

    Reference: Blanchard et al. "Machine learning with adversaries: Byzantine
    tolerant gradient descent." NIPS 2017.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MultiKrum defense.

        Args:
            config: Configuration with keys:
                - num_malicious: Estimated number of malicious clients
                - num_selection: Number of updates to select (default: n_clients - 2f - 1)
        """
        super().__init__(config)
        self.num_malicious = config.get("num_malicious", 0)
        self.num_selection = config.get("num_selection", None)

        # Use Krum defense with multi_krum=True
        self._krum = KrumDefense({
            "num_malicious": self.num_malicious,
            "multi_krum": True,
            "num_selection": self.num_selection,
        })

    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate updates using MultiKrum.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters
        """
        return self._krum.defend(updates)


class BulyanDefense(BaseDefense):
    """
    Bulyan aggregation - combines Krum selection with coordinate-wise median.

    More robust than Krum alone, especially against colluding attackers.
    First selects (n - 4f) updates using Krum, then applies coordinate-wise
    trimming and median.

    Reference: Mhamdi et al. "The hidden vulnerability of distributed learning
    in Byzantium." ICML 2018.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Bulyan defense.

        Args:
            config: Configuration with keys:
                - num_malicious: Number of malicious clients to tolerate
        """
        super().__init__(config)
        self.num_malicious = config.get("num_malicious", 0)

    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate updates using Bulyan.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters
        """
        params = self._extract_updates(updates)
        n_clients = len(params)
        f = self.num_malicious if self.num_malicious > 0 else 0

        # Step 1: Select n - 3f updates using distance-based selection
        n_select = n_clients - 3 * f
        if n_select <= 0:
            # Fallback to simple median
            return np.median(params, axis=0)

        # Compute scores similar to Krum
        scores = self._compute_scores(params)
        selected_indices = np.argsort(scores)[:n_select]
        selected_params = params[selected_indices]

        # Step 2: Apply coordinate-wise trimmed median
        # For each dimension, remove 2f extremes and take median
        theta = n_select - 2 * f
        if theta <= 0:
            return np.median(selected_params, axis=0)

        result = np.zeros(selected_params.shape[1])

        for dim in range(selected_params.shape[1]):
            dim_values = selected_params[:, dim]
            sorted_idx = np.argsort(dim_values)
            trimmed = dim_values[sorted_idx[f:n_select - f]]
            result[dim] = np.median(trimmed)

        return result

    def _compute_scores(self, params: NDArray) -> NDArray:
        """
        Compute distance-based scores for selection.

        Args:
            params: Array of shape (n_clients, n_params)

        Returns:
            Array of scores for each client
        """
        n_clients = len(params)
        f = self.num_malicious if self.num_malicious > 0 else 0
        num_closest = n_clients - f - 2

        scores = np.zeros(n_clients)

        for i in range(n_clients):
            distances = np.sum((params - params[i]) ** 2, axis=1)
            scores[i] = np.sum(np.sort(distances)[:num_closest])

        return scores


def create_defense(defense_type: str, config: Dict[str, Any]) -> BaseDefense:
    """
    Factory function to create a defense instance.

    Args:
        defense_type: Type of defense ('fedavg', 'median', 'trimmed_mean', 'krum', 'multikrum', 'bulyan')
        config: Configuration dictionary

    Returns:
        Defense instance

    Raises:
        ValueError: If defense_type is unknown
    """
    defense_map = {
        "fedavg": FedAvgDefense,
        "median": MedianDefense,
        "trimmed_mean": TrimmedMeanDefense,
        "krum": KrumDefense,
        "multikrum": MultiKrumDefense,
        "bulyan": BulyanDefense,
    }

    defense_class = defense_map.get(defense_type.lower())
    if defense_class is None:
        raise ValueError(f"Unknown defense type: {defense_type}")

    return defense_class(config)
