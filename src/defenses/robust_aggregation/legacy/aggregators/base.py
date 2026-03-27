"""
Base class for Byzantine-robust aggregators.

All robust aggregators must inherit from RobustAggregator and implement
the aggregate() method.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import torch


class RobustAggregator(ABC):
    """
    Abstract base class for Byzantine-robust aggregation methods.

    All aggregators must:
    1. Handle a list of model updates from clients
    2. Know the number of potential Byzantine attackers
    3. Return a single aggregated model update
    """

    @abstractmethod
    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        num_attackers: int
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates from multiple clients, resilient to Byzantine attacks.

        Args:
            updates: List of model updates, where each update is a dict mapping
                    parameter names to torch.Tensor values
            num_attackers: Maximum number of Byzantine (malicious) clients

        Returns:
            Aggregated model update as a dict with same keys as input updates

        Raises:
            ValueError: If insufficient updates for robust aggregation
        """
        pass

    def _validate_updates(self, updates: List[Dict[str, torch.Tensor]]) -> None:
        """Validate that all updates have matching structure."""
        if len(updates) == 0:
            raise ValueError("Cannot aggregate empty list of updates")

        # Get reference structure from first update
        ref_keys = set(updates[0].keys())
        ref_shapes = {k: v.shape for k, v in updates[0].items()}

        for i, update in enumerate(updates):
            if set(update.keys()) != ref_keys:
                raise ValueError(
                    f"Update {i} has mismatched keys. "
                    f"Expected {ref_keys}, got {set(update.keys())}"
                )
            for key in ref_keys:
                if update[key].shape != ref_shapes[key]:
                    raise ValueError(
                        f"Update {i}, parameter '{key}' has mismatched shape. "
                        f"Expected {ref_shapes[key]}, got {update[key].shape}"
                    )

    def _stack_updates(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Stack all client updates into batched tensors.

        Returns:
            Dict mapping parameter names to tensors of shape (num_clients, *param_shape)
        """
        stacked = {}
        for key in updates[0].keys():
            stacked[key] = torch.stack([u[key] for u in updates])
        return stacked

    @property
    def name(self) -> str:
        """Return the name of this aggregation method."""
        return self.__class__.__name__
