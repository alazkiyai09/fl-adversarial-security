"""
Base aggregator interface for federated learning defenses.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from flwr.common import Parameters, FitRes
from flwr.server.client_proxy import ClientProxy


class BaseAggregator(ABC):
    """Abstract base class for all aggregation strategies."""

    def __init__(self, **kwargs):
        """Initialize aggregator with hyperparameters."""
        self.params = kwargs
        self.history: Dict[str, Any] = {}

    @abstractmethod
    def aggregate(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Parameters:
        """
        Aggregate client updates into global parameters.

        Args:
            results: List of (client_proxy, fit_result) tuples
                    containing client updates

        Returns:
            Aggregated global parameters
        """
        pass

    def reset_history(self) -> None:
        """Reset aggregation history."""
        self.history = {}

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregation metrics for analysis."""
        return self.history
