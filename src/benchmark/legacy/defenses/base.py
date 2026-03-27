"""
Abstract base class for all defense implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from numpy.typing import NDArray
import numpy as np


class BaseDefense(ABC):
    """
    Abstract base class for federated learning defenses.

    All defenses must inherit from this class and implement the defend method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize defense with configuration.

        Args:
            config: Defense-specific configuration parameters
        """
        self.config = config
        self.name = self.__class__.__name__
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset defense state between experiments.
        Override in subclasses that maintain state across rounds.
        """
        pass

    @abstractmethod
    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate potentially malicious updates into robust global model.

        Args:
            updates: List of (client_id, parameters) tuples from all clients

        Returns:
            Aggregated global model parameters as numpy array
        """
        pass

    def get_detection_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get detection metrics if defense performs malicious client identification.

        Returns:
            Dictionary with detection metrics (precision, recall, f1, accuracy)
            or None if defense doesn't perform detection
        """
        return None

    def get_info(self) -> Dict[str, Any]:
        """
        Get defense information and configuration.

        Returns:
            Dictionary with defense metadata
        """
        return {
            "name": self.name,
            "config": self.config,
        }

    def _extract_updates(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Extract parameter arrays from update tuples.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Stacked parameters array of shape (n_clients, n_params)
        """
        return np.array([params for _, params in updates])

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"


class DefenseResult:
    """
    Container for defense aggregation results.
    """

    def __init__(
        self,
        aggregated_params: NDArray,
        defense_metrics: Optional[Dict[str, float]] = None,
        detected_malicious: Optional[List[int]] = None,
        execution_time: float = 0.0,
    ):
        """
        Initialize defense result.

        Args:
            aggregated_params: Final aggregated parameters
            defense_metrics: Optional detection metrics (precision, recall, etc.)
            detected_malicious: List of detected malicious client IDs
            execution_time: Defense execution time in seconds
        """
        self.aggregated_params = aggregated_params
        self.defense_metrics = defense_metrics or {}
        self.detected_malicious = detected_malicious or []
        self.execution_time = execution_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "aggregated_params": self.aggregated_params,
            "defense_metrics": self.defense_metrics,
            "detected_malicious": self.detected_malicious,
            "execution_time": self.execution_time,
        }
