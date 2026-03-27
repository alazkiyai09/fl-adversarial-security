"""
Final aggregation of unmasked model updates.
"""

import torch
from typing import List, Dict, Optional
import copy


def sum_updates(updates: List[torch.Tensor]) -> torch.Tensor:
    """
    Sum multiple model updates.

    Args:
        updates: List of update tensors

    Returns:
        Sum of all updates

    Raises:
        ValueError: If updates list is empty or shapes don't match
    """
    if not updates:
        raise ValueError("Cannot sum empty list of updates")

    # Verify all updates have same shape
    shape = updates[0].shape
    dtype = updates[0].dtype

    for update in updates:
        if update.shape != shape:
            raise ValueError(f"Shape mismatch: {shape} vs {update.shape}")
        if update.dtype != dtype:
            raise ValueError(f"Dtype mismatch: {dtype} vs {update.dtype}")

    # Sum all updates
    total = torch.zeros(shape, dtype=dtype)
    for update in updates:
        total = total + update

    return total


def compute_average(updates: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute average of multiple model updates.

    Args:
        updates: List of update tensors

    Returns:
        Average of all updates
    """
    total = sum_updates(updates)
    average = total / len(updates)
    return average


class SecureAggregator:
    """
    Handles secure aggregation of model updates.

    Server-side component that:
    1. Receives masked updates from clients
    2. Collects and verifies mask shares
    3. Reconstructs masks from dropouts
    4. Cancels all masks
    5. Computes final aggregate
    """

    def __init__(
        self,
        num_clients: int,
        model_shape: torch.Size,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the secure aggregator.

        Args:
            num_clients: Total number of clients in the system
            model_shape: Shape of model update tensors
            dtype: Data type of model parameters
        """
        self.num_clients = num_clients
        self.model_shape = model_shape
        self.dtype = dtype

        # Storage
        self.masked_updates: Dict[int, torch.Tensor] = {}
        self.mask_shares: Dict[int, List] = {}  # client_id -> list of shares
        self.reconstructed_masks: Dict[int, torch.Tensor] = {}

        # Metrics
        self.communication_cost = {
            'masked_updates_bytes': 0,
            'mask_shares_bytes': 0,
            'total_bytes': 0
        }

    def receive_masked_update(self, client_id: int, update: torch.Tensor) -> None:
        """
        Receive a masked update from a client.

        Args:
            client_id: Client identifier
            update: Masked model update tensor
        """
        if update.shape != self.model_shape:
            raise ValueError(f"Update shape {update.shape} doesn't match expected {self.model_shape}")

        self.masked_updates[client_id] = update.clone()

        # Track communication cost
        self.communication_cost['masked_updates_bytes'] += update.element_size() * update.numel()

    def receive_mask_shares(self, client_id: int, shares: List) -> None:
        """
        Receive mask shares from a client.

        Args:
            client_id: Client identifier
            shares: List of (target_id, share_index, share_value) tuples
        """
        self.mask_shares[client_id] = shares

        # Track communication cost (rough estimate)
        self.communication_cost['mask_shares_bytes'] += len(shares) * 16  # 16 bytes per share

    def compute_aggregate(self, active_clients: Optional[List[int]] = None) -> torch.Tensor:
        """
        Compute the final aggregate of unmasked updates.

        Args:
            active_clients: List of active client IDs (None = all clients)

        Returns:
            Aggregated model update tensor
        """
        if active_clients is None:
            active_clients = list(self.masked_updates.keys())

        if len(active_clients) == 0:
            raise ValueError("No active clients to aggregate")

        # Start with sum of masked updates
        total = torch.zeros(self.model_shape, dtype=self.dtype)
        for client_id in active_clients:
            if client_id in self.masked_updates:
                total = total + self.masked_updates[client_id]

        # Subtract all reconstructed masks (they should cancel to zero)
        for mask in self.reconstructed_masks.values():
            total = total - mask

        return total

    def verify_mask_cancellation(self) -> bool:
        """
        Verify that all masks cancel out correctly.

        Returns:
            True if mask sum is approximately zero
        """
        if not self.reconstructed_masks:
            return True  # No masks to cancel

        total_mask = torch.zeros(self.model_shape, dtype=self.dtype)
        for mask in self.reconstructed_masks.values():
            total_mask = total_mask + mask

        tolerance = 1e-6
        return torch.all(torch.abs(total_mask) < tolerance).item()

    def get_communication_cost(self) -> Dict[str, int]:
        """
        Get total communication cost.

        Returns:
            Dictionary with cost breakdown
        """
        self.communication_cost['total_bytes'] = (
            self.communication_cost['masked_updates_bytes'] +
            self.communication_cost['mask_shares_bytes']
        )
        return copy.deepcopy(self.communication_cost)

    def reset(self) -> None:
        """Reset aggregator state for next round."""
        self.masked_updates.clear()
        self.mask_shares.clear()
        self.reconstructed_masks.clear()
        self.communication_cost = {
            'masked_updates_bytes': 0,
            'mask_shares_bytes': 0,
            'total_bytes': 0
        }
