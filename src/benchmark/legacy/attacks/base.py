"""
Abstract base class for all attack implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader


class BaseAttack(ABC):
    """
    Abstract base class for federated learning attacks.

    All attacks must inherit from this class and implement the apply_attack method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize attack with configuration.

        Args:
            config: Attack-specific configuration parameters
        """
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def apply_attack(
        self,
        parameters: NDArray,
        local_data: Optional[DataLoader] = None,
        client_id: int = 0,
        global_model: Optional[torch.nn.Module] = None,
    ) -> NDArray:
        """
        Apply attack to client's local model parameters.

        Args:
            parameters: Client's local model parameters as numpy array
            local_data: Optional local training data for data-dependent attacks
            client_id: ID of the attacking client
            global_model: Optional global model reference

        Returns:
            Poisoned parameters as numpy array
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get attack information and configuration.

        Returns:
            Dictionary with attack metadata
        """
        return {
            "name": self.name,
            "config": self.config,
        }

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
