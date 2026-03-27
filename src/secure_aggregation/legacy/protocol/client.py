"""
Client-side protocol for secure aggregation.

Implements the client's role in the secure aggregation protocol:
1. Pairwise key agreement
2. Mask generation from shared secrets
3. Mask sharing with other clients
4. Masked update submission
5. Dropout recovery participation
"""

import random
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..crypto import (
    generate_dh_keypair,
    compute_shared_secret,
    split_secret,
    generate_mask_from_seed,
    pairwise_key_agreement
)
from ..communication import CommunicationChannel, MessageType
from ..aggregation import apply_mask


@dataclass
class ClientState:
    """State of a client during the protocol."""

    client_id: int
    model_update: torch.Tensor
    public_key: int = 0
    private_key: int = 0
    shared_secrets: Dict[int, int] = None
    pairwise_masks: Dict[int, torch.Tensor] = None
    my_mask: torch.Tensor = None
    my_mask_seed: int = 0
    mask_shares: List[Tuple[int, int, int]] = None

    def __post_init__(self):
        if self.shared_secrets is None:
            self.shared_secrets = {}
        if self.pairwise_masks is None:
            self.pairwise_masks = {}
        if self.my_mask is None:
            self.my_mask = torch.zeros_like(self.model_update)
        if self.mask_shares is None:
            self.mask_shares = []


class SecureAggregationClient:
    """
    Client in the secure aggregation protocol.

    Each client:
    - Performs DH key agreement with all other clients
    - Generates a random mask
    - Secret shares the mask seed
    - Submits masked update to server
    - Participates in dropout recovery
    """

    def __init__(
        self,
        client_id: int,
        model_update: torch.Tensor,
        config: Dict[str, Any]
    ):
        """
        Initialize a secure aggregation client.

        Args:
            client_id: Unique client identifier
            model_update: This client's model update
            config: Configuration dictionary
        """
        self.client_id = client_id
        self.model_update = model_update
        self.config = config

        # Cryptographic parameters
        self.dh_prime = config.get('dh_prime', 2**127 - 1)  # Default prime for demo
        self.dh_generator = config.get('dh_generator', 2)
        self.sharing_prime = config.get('secret_sharing_prime', 2**127 - 1)

        # Threshold parameter
        num_clients = config.get('num_clients', 10)
        threshold_ratio = config.get('threshold_ratio', 0.7)
        self.threshold = int(num_clients * threshold_ratio)

        # Protocol state
        self.state = ClientState(
            client_id=client_id,
            model_update=model_update
        )

        # Communication
        self.channel: Optional[CommunicationChannel] = None
        self.server_id = -1  # Convention: server has ID -1

    def setup_pairwise_keys(self, all_client_ids: List[int]) -> None:
        """
        Perform pairwise Diffie-Hellman key agreement.

        Args:
            all_client_ids: List of all client IDs in the system
        """
        # Generate DH key pair
        self.state.private_key, self.state.public_key = generate_dh_keypair(
            self.dh_prime, self.dh_generator
        )

        # Compute shared secrets with all other clients
        # In simulation, we use pairwise_key_agreement which simulates
        # the actual key exchange
        self.state.shared_secrets = pairwise_key_agreement(
            self.client_id,
            all_client_ids,
            self.dh_prime,
            self.dh_generator
        )

    def generate_masks_and_seeds(self) -> Tuple[torch.Tensor, int]:
        """
        Generate random mask and its seed.

        Returns:
            Tuple of (mask_tensor, seed)
        """
        # Generate random seed
        self.state.my_mask_seed = random.randint(1, 2**32 - 1)

        # Generate mask from seed
        self.state.my_mask = generate_mask_from_seed(
            self.state.my_mask_seed,
            self.model_update.shape,
            self.model_update.dtype
        )

        return self.state.my_mask, self.state.my_mask_seed

    def create_mask_shares(self, seed: int, num_clients: int) -> List[Tuple[int, int, int]]:
        """
        Create secret shares of the mask seed.

        Args:
            seed: The seed to share
            num_clients: Total number of clients

        Returns:
            List of (target_client_id, share_index, share_value) tuples
        """
        # Split seed into shares
        # Note: split_secret returns 1-indexed shares (x=1,2,...,n)
        # because f(0) = secret (the constant term)
        shares = split_secret(seed, self.threshold, num_clients, self.sharing_prime)

        # Each share is (share_index, share_value) where share_index is 1-indexed
        # We need to map to 0-indexed client IDs
        # Share 1 goes to client 0, share 2 goes to client 1, etc.
        mask_shares = []
        for share_index, share_value in shares:
            target_client_id = share_index - 1  # Convert to 0-indexed client ID
            mask_shares.append((target_client_id, share_index, share_value))

        self.state.mask_shares = mask_shares
        return mask_shares

    def submit_masked_update(self) -> torch.Tensor:
        """
        Apply mask and submit masked update to server.

        Returns:
            The masked update tensor
        """
        # Apply mask to update
        masked_update = apply_mask(self.model_update, self.state.my_mask)

        return masked_update

    def compute_pairwise_mask_contribution(self) -> torch.Tensor:
        """
        Compute the total mask contribution from pairwise shared secrets.

        Each pair of clients shares a secret, which generates a mask.
        A client's total mask is the sum of its own mask plus all pairwise masks.

        Returns:
            Combined mask from all pairwise secrets
        """
        combined_mask = torch.zeros_like(self.model_update)

        for peer_id, shared_secret in self.state.shared_secrets.items():
            # Generate mask from shared secret
            # Use ordered pair to ensure consistency
            ordered_pair = tuple(sorted([self.client_id, peer_id]))
            pair_seed = hash(ordered_pair + (shared_secret,)) % (2**32)

            pair_mask = generate_mask_from_seed(
                pair_seed,
                self.model_update.shape,
                self.model_update.dtype
            )

            # Add to combined mask
            # Note: One client adds, other subtracts (they cancel pairwise)
            if self.client_id < peer_id:
                combined_mask = combined_mask + pair_mask
            else:
                combined_mask = combined_mask - pair_mask

        return combined_mask

    def get_final_mask(self) -> torch.Tensor:
        """
        Get the final mask to apply (own mask + pairwise contributions).

        Returns:
            Final mask tensor
        """
        pairwise_mask = self.compute_pairwise_mask_contribution()
        final_mask = self.state.my_mask + pairwise_mask
        return final_mask

    def submit_masked_update_final(self) -> torch.Tensor:
        """
        Apply final mask and submit masked update.

        Returns:
            Masked update tensor
        """
        final_mask = self.get_final_mask()
        masked_update = apply_mask(self.model_update, final_mask)
        return masked_update

    def respond_to_dropout(
        self,
        dead_client_ids: List[int],
        all_client_ids: List[int]
    ) -> List[Tuple[int, int, int]]:
        """
        Respond to client dropout by submitting relevant shares.

        For each dead client, we need to submit our share (the share
        corresponding to our client_id) to help reconstruct their seed.

        Args:
            dead_client_ids: List of clients that dropped out
            all_client_ids: List of all client IDs

        Returns:
            List of (dead_client_id, share_index, share_value) tuples
        """
        relevant_shares = []

        # Find the share corresponding to our client_id
        # Our client_id is 0-indexed, share_index is 1-indexed
        my_share_index = self.client_id + 1  # Convert to 1-indexed
        my_share_value = None

        # In our mask_shares, find the entry where share_index == our client_id + 1
        for target_id, share_idx, share_val in self.state.mask_shares:
            if share_idx == my_share_index:
                my_share_value = share_val
                break

        if my_share_value is None:
            # We don't have a share for ourselves (shouldn't happen)
            return []

        # For each dead client, submit our share
        # In the simplified protocol, all clients share their seed the same way
        # So we submit our share for each dead client
        for dead_id in dead_client_ids:
            relevant_shares.append((dead_id, my_share_index, my_share_value))

        return relevant_shares

    def receive_share_from_peer(
        self,
        peer_id: int,
        share: Tuple[int, int]
    ) -> None:
        """
        Receive a share from another client during dropout recovery.

        Args:
            peer_id: Client sending the share
            share: (share_index, share_value) tuple
        """
        # Store share for later reconstruction
        # In full implementation, this would be stored per dead client
        pass

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of client state for debugging.

        Returns:
            Dictionary with state information
        """
        return {
            'client_id': self.client_id,
            'has_model_update': self.model_update is not None,
            'update_shape': self.model_update.shape if self.model_update is not None else None,
            'num_shared_secrets': len(self.state.shared_secrets),
            'has_mask': self.state.my_mask is not None,
            'mask_seed': self.state.my_mask_seed,
            'num_mask_shares': len(self.state.mask_shares),
            'threshold': self.threshold
        }
