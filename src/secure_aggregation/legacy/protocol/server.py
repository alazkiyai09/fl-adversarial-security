"""
Server-side protocol for secure aggregation.

Implements the server's role in the secure aggregation protocol:
1. Coordinate key agreement
2. Receive masked updates from clients
3. Handle client dropouts
4. Coordinate mask reconstruction
5. Compute final aggregate
"""

import time
import torch
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum

from ..crypto import reconstruct_secret
from ..communication import CommunicationChannel, MessageType
from ..aggregation import SecureAggregator


class ServerState(Enum):
    """Server protocol states."""
    IDLE = "idle"
    KEY_AGREEMENT = "key_agreement"
    COLLECTING_UPDATES = "collecting_updates"
    COLLECTING_SHARES = "collecting_shares"
    DROPOUT_RECOVERY = "dropout_recovery"
    COMPUTING_AGGREGATE = "computing_aggregate"
    COMPLETE = "complete"


@dataclass
class ClientInfo:
    """Information about a client."""
    client_id: int
    is_active: bool = True
    submitted_update: bool = False
    submitted_shares: bool = False


class SecureAggregationServer:
    """
    Server in the secure aggregation protocol.

    The server:
    - Coordinates the protocol phases
    - Receives masked updates from clients
    - Detects and handles client dropouts
    - Coordinates mask reconstruction
    - Computes the final aggregate
    - Only sees the sum of updates, not individual values
    """

    def __init__(
        self,
        num_clients: int,
        model_shape: torch.Size,
        config: Dict[str, Any]
    ):
        """
        Initialize the secure aggregation server.

        Args:
            num_clients: Total number of clients
            model_shape: Shape of model update tensors
            config: Configuration dictionary
        """
        self.num_clients = num_clients
        self.model_shape = model_shape
        self.config = config

        # Cryptographic parameters
        self.sharing_prime = config.get('secret_sharing_prime', 2**127 - 1)
        self.threshold_ratio = config.get('threshold_ratio', 0.7)
        self.threshold = int(num_clients * self.threshold_ratio)
        self.dropout_tolerance = config.get('dropout_tolerance', 0.3)
        self.timeout = config.get('timeout_seconds', 30.0)

        # Client management
        self.clients: Dict[int, ClientInfo] = {}
        for i in range(num_clients):
            self.clients[i] = ClientInfo(client_id=i)

        # Protocol state
        self.state = ServerState.IDLE
        self.current_round = 0

        # Aggregation
        self.aggregator = SecureAggregator(num_clients, model_shape)

        # Mask seed shares storage
        self.seed_shares: Dict[int, List[Tuple[int, int, int]]] = {}  # client_id -> shares
        self.reconstructed_seeds: Dict[int, int] = {}  # client_id -> seed

        # Communication
        self.channel: Optional[CommunicationChannel] = None

        # Metrics
        self.metrics = {
            'rounds_completed': 0,
            'total_dropouts': 0,
            'successful_reconstructions': 0,
            'failed_reconstructions': 0,
            'communication_overhead': 0
        }

    def start_round(self, round_num: int) -> None:
        """
        Start a new aggregation round.

        Args:
            round_num: Round number
        """
        self.current_round = round_num
        self.state = ServerState.COLLECTING_UPDATES
        self.aggregator.reset()
        self.seed_shares.clear()
        self.reconstructed_seeds.clear()

        # Reset client submission status
        for client in self.clients.values():
            client.submitted_update = False
            client.submitted_shares = False

    def receive_masked_update(self, client_id: int, update: torch.Tensor) -> bool:
        """
        Receive a masked update from a client.

        Args:
            client_id: Client identifier
            update: Masked model update

        Returns:
            True if received successfully
        """
        if client_id not in self.clients:
            return False

        if not self.clients[client_id].is_active:
            return False

        self.aggregator.receive_masked_update(client_id, update)
        self.clients[client_id].submitted_update = True

        return True

    def receive_seed_shares(
        self,
        client_id: int,
        shares: List[Tuple[int, int, int]]
    ) -> bool:
        """
        Receive seed shares from a client.

        Args:
            client_id: Client identifier
            shares: List of (target_id, share_index, share_value) tuples

        Returns:
            True if received successfully
        """
        if client_id not in self.clients:
            return False

        self.seed_shares[client_id] = shares
        self.clients[client_id].submitted_shares = True

        # Also pass to aggregator for reconstruction
        self.aggregator.receive_mask_shares(client_id, shares)

        return True

    def get_active_clients(self) -> List[int]:
        """Get list of active client IDs."""
        return [
            cid for cid, info in self.clients.items()
            if info.is_active
        ]

    def get_pending_clients(self) -> List[int]:
        """Get list of clients that haven't submitted updates."""
        return [
            cid for cid, info in self.clients.items()
            if info.is_active and not info.submitted_update
        ]

    def detect_dead_clients(
        self,
        timeout: float = 5.0
    ) -> List[int]:
        """
        Detect clients that haven't submitted within timeout.

        Args:
            timeout: Timeout in seconds

        Returns:
            List of dead client IDs
        """
        start_time = time.time()
        dead_clients = []

        # In simulation, we just check who hasn't submitted
        # In real system, this would use actual timeouts
        for client_id, info in self.clients.items():
            if info.is_active and not info.submitted_update:
                # Check timeout (simulated)
                if time.time() - start_time > timeout:
                    dead_clients.append(client_id)

        return dead_clients

    def mark_clients_dead(self, client_ids: List[int]) -> None:
        """
        Mark clients as dead (dropped out).

        Args:
            client_ids: List of client IDs to mark dead
        """
        for client_id in client_ids:
            if client_id in self.clients:
                self.clients[client_id].is_active = False
                self.metrics['total_dropouts'] += 1

    def validate_threshold_sufficient(
        self,
        num_active: int,
        num_dead: int
    ) -> bool:
        """
        Check if we have enough clients to recover from dropouts.

        Args:
            num_active: Number of active clients
            num_dead: Number of dead clients

        Returns:
            True if threshold is sufficient for reconstruction
        """
        # Need at least threshold active clients for reconstruction
        return num_active >= self.threshold

    def reconstruct_dead_client_seeds(self, dead_client_ids: List[int]) -> Dict[int, int]:
        """
        Reconstruct mask seeds for dead clients using shares.

        Args:
            dead_client_ids: List of dead client IDs

        Returns:
            Dictionary mapping client_id -> reconstructed_seed
        """
        reconstructed = {}

        for dead_id in dead_client_ids:
            # Collect shares for this dead client
            shares = []
            for client_id, share_list in self.seed_shares.items():
                if self.clients[client_id].is_active:
                    # Find shares meant for this dead client
                    for target_id, share_idx, share_val in share_list:
                        if target_id == dead_id:
                            shares.append((share_idx, share_val))

            # Need at least threshold shares for reconstruction
            if len(shares) >= self.threshold:
                try:
                    seed = reconstruct_secret(shares, self.sharing_prime)
                    reconstructed[dead_id] = seed
                    self.metrics['successful_reconstructions'] += 1
                except Exception as e:
                    self.metrics['failed_reconstructions'] += 1
                    print(f"Failed to reconstruct seed for client {dead_id}: {e}")
            else:
                self.metrics['failed_reconstructions'] += 1
                print(f"Insufficient shares for client {dead_id}: {len(shares)} < {self.threshold}")

        return reconstructed

    def can_proceed(self) -> bool:
        """
        Check if we can proceed to compute aggregate.

        Returns:
            True if we have sufficient clients
        """
        active_clients = self.get_active_clients()
        num_active = len(active_clients)

        # Need at least threshold clients
        return num_active >= self.threshold

    def compute_aggregate(self) -> Optional[torch.Tensor]:
        """
        Compute the final aggregate.

        Returns:
            Aggregated model update, or None if insufficient clients
        """
        if not self.can_proceed():
            print(f"Cannot proceed: insufficient active clients")
            return None

        active_clients = self.get_active_clients()
        aggregate = self.aggregator.compute_aggregate(active_clients)

        self.state = ServerState.COMPLETE
        self.metrics['rounds_completed'] += 1

        return aggregate

    def verify_mask_cancellation(self) -> bool:
        """
        Verify that all masks cancel correctly.

        Returns:
            True if masks sum to zero
        """
        return self.aggregator.verify_mask_cancellation()

    def get_communication_cost(self) -> Dict[str, int]:
        """
        Get communication cost for this round.

        Returns:
            Dictionary with cost breakdown
        """
        return self.aggregator.get_communication_cost()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get server metrics.

        Returns:
            Dictionary with metrics
        """
        active_clients = self.get_active_clients()

        return {
            **self.metrics,
            'current_state': self.state.value,
            'active_clients': len(active_clients),
            'dead_clients': self.num_clients - len(active_clients),
            'dropout_rate': (self.num_clients - len(active_clients)) / self.num_clients
        }

    def reset_round(self) -> None:
        """Reset server state for next round."""
        self.state = ServerState.IDLE
        self.aggregator.reset()
        self.seed_shares.clear()
        self.reconstructed_seeds.clear()

        for client in self.clients.values():
            client.submitted_update = False
            client.submitted_shares = False
