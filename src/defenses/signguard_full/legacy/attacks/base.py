"""Base class for attack implementations."""

from abc import ABC, abstractmethod
from typing import List
import torch

from src.defenses.signguard_full.legacy.core.types import ModelUpdate
from torch.utils.data import DataLoader


class Attack(ABC):
    """Abstract base class for FL attacks."""

    @abstractmethod
    def execute(
        self,
        client_id: str,
        global_model: dict[str, torch.Tensor],
        train_loader: DataLoader | None = None,
    ) -> ModelUpdate:
        """Execute attack and create malicious update.

        Args:
            client_id: Client identifier
            global_model: Current global model
            train_loader: Optional training data for data poisoning

        Returns:
            Malicious model update
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get attack name.

        Returns:
            Attack name string
        """
        pass
