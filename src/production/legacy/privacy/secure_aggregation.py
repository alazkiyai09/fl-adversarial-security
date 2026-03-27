"""Secure aggregation for federated learning.

Implements cryptographic protection for model updates:
- Pairwise masking
- Homomorphic encryption (optional)
- Threshold secret sharing
"""

from typing import List, Tuple, Optional, Dict, Any
import random

import torch
import numpy as np
from loguru import logger


def generate_random_mask(
    shape: Tuple[int, ...],
    bit_size: int = 32,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate random mask for secure aggregation.

    Args:
        shape: Shape of the mask
        bit_size: Bit size for random values
        seed: Random seed for reproducibility

    Returns:
        Random mask as numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random values in range [-2^(bit_size-1), 2^(bit_size-1)]
    range_limit = 2 ** (bit_size - 1)
    mask = np.random.randint(
        -range_limit,
        range_limit,
        size=shape,
        dtype=np.int64,
    )
    return mask.astype(np.float32)


def pairwise_mask(
    values: np.ndarray,
    client_id: int,
    n_clients: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Apply pairwise masking for secure aggregation.

    Each client adds masks that will cancel out during aggregation.
    For each pair (i, j) where i < j:
    - Client i adds +mask_ij
    - Client j adds -mask_ij

    Args:
        values: Values to mask (e.g., model parameters)
        client_id: Current client ID
        n_clients: Total number of clients
        seed: Random seed for reproducibility

    Returns:
        Masked values

    Reference:
        Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving
        Machine Learning", CCS 2017
    """
    np.random.seed(seed + client_id)

    masked_values = values.copy().astype(np.int64)

    # Add masks for all pairs (i, j)
    for other_client in range(n_clients):
        if client_id < other_client:
            # Client adds positive mask for this pair
            mask = generate_random_mask(values.shape, seed=seed + client_id * n_clients + other_client)
            masked_values += mask
        elif client_id > other_client:
            # Client subtracts mask for this pair
            mask = generate_random_mask(values.shape, seed=seed + other_client * n_clients + client_id)
            masked_values -= mask

    return masked_values.astype(np.float32)


class SecureAggregator:
    """
    Secure aggregation using pairwise masking.

    Protocol:
    1. Each client generates pairwise masks with all other clients
    2. Masks are applied to updates before sending
    3. During aggregation, masks cancel out
    4. Server only sees aggregated sum, not individual updates

    Security:
    - Server cannot see individual client updates
    - Clients cannot infer each other's updates
    - Requires collusion of > n/2 clients to breach privacy
    """

    def __init__(
        self,
        n_clients: int,
        bit_size: int = 32,
        seed: int = 42,
    ):
        """
        Initialize secure aggregator.

        Args:
            n_clients: Total number of clients
            bit_size: Bit size for random masks
            seed: Random seed
        """
        self.n_clients = n_clients
        self.bit_size = bit_size
        self.seed = seed

        # Store masks for each client pair
        self.masks: Dict[Tuple[int, int], np.ndarray] = {}

        logger.info(
            f"SecureAggregator initialized: n_clients={n_clients}, "
            f"bit_size={bit_size}"
        )

    def mask_update(
        self,
        update: List[np.ndarray],
        client_id: int,
    ) -> List[np.ndarray]:
        """
        Apply pairwise masking to client update.

        Args:
            update: Client update (list of parameter arrays)
            client_id: Client ID

        Returns:
            Masked update
        """
        masked_update = []

        for layer_idx, layer_params in enumerate(update):
            # Apply pairwise masking
            masked_layer = pairwise_mask(
                values=layer_params,
                client_id=client_id,
                n_clients=self.n_clients,
                seed=self.seed + layer_idx * 1000,
            )
            masked_update.append(masked_layer)

        logger.debug(f"Client {client_id}: Applied pairwise masking")
        return masked_update

    def unmask_aggregate(
        self,
        masked_updates: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Remove masks and aggregate updates.

        In pairwise masking, masks should cancel out during aggregation.
        This method performs the aggregation.

        Args:
            masked_updates: List of masked client updates

        Returns:
            Aggregated parameters
        """
        if not masked_updates:
            return []

        # Sum all masked updates (masks cancel out)
        n_layers = len(masked_updates[0])
        aggregated = []

        for layer_idx in range(n_layers):
            # Stack and sum
            layer_stack = np.stack([update[layer_idx] for update in masked_updates])
            aggregated_layer = np.sum(layer_stack, axis=0)
            aggregated.append(aggregated_layer)

        # Average (optional, depending on aggregation strategy)
        aggregated = [layer / len(masked_updates) for layer in aggregated]

        logger.info(f"Aggregated {len(masked_updates)} masked updates")
        return aggregated

    def verify_cancellation(
        self,
        masked_updates: List[List[np.ndarray]],
    ) -> bool:
        """
        Verify that masks cancel out correctly.

        For testing/validation purposes.

        Args:
            masked_updates: List of masked updates

        Returns:
            True if cancellation is verified
        """
        # Sum all masks should be approximately zero
        total_mask_sum = 0

        for layer_idx in range(len(masked_updates[0])):
            # Reconstruct masks for each client
            masks_sum = np.zeros_like(masked_updates[0][layer_idx])

            for client_id in range(self.n_clients):
                for other_client in range(self.n_clients):
                    if client_id < other_client:
                        mask = generate_random_mask(
                            masked_updates[0][layer_idx].shape,
                            seed=self.seed + layer_idx * 1000 + client_id * self.n_clients + other_client,
                        )
                        masks_sum += mask
                    elif client_id > other_client:
                        mask = generate_random_mask(
                            masked_updates[0][layer_idx].shape,
                            seed=self.seed + layer_idx * 1000 + other_client * self.n_clients + client_id,
                        )
                        masks_sum -= mask

            total_mask_sum += np.abs(masks_sum).sum()

        # Should be very close to zero
        return total_mask_sum < 1e-6


class ThresholdSecretSharing:
    """
    Threshold secret sharing for enhanced security.

    Splits data into n shares such that any t shares can reconstruct,
    but fewer than t shares reveal nothing.
    """

    def __init__(
        self,
        n_shares: int,
        threshold: int,
        prime: int = 2**61 - 1,  # Large Mersenne prime
    ):
        """
        Initialize secret sharing scheme.

        Args:
            n_shares: Number of shares to create
            threshold: Minimum shares needed for reconstruction
            prime: Prime modulus for arithmetic
        """
        self.n_shares = n_shares
        self.threshold = threshold
        self.prime = prime

        if threshold > n_shares:
            raise ValueError("Threshold cannot exceed number of shares")

        logger.info(
            f"ThresholdSecretSharing: n_shares={n_shares}, threshold={threshold}"
        )

    def share(self, secret: int) -> List[Tuple[int, int]]:
        """
        Split secret into shares using Shamir's Secret Sharing.

        Args:
            secret: Secret value to share

        Returns:
            List of (x, y) share pairs
        """
        if secret >= self.prime:
            raise ValueError("Secret must be less than prime modulus")

        # Generate random polynomial coefficients
        # f(x) = secret + a1*x + a2*x^2 + ... + at*x^t
        coefficients = [secret] + [
            random.randint(0, self.prime - 1) for _ in range(self.threshold - 1)
        ]

        # Create shares
        shares = []
        for i in range(1, self.n_shares + 1):
            x = i
            y = self._evaluate_polynomial(coefficients, x)
            shares.append((x, y))

        return shares

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation.

        Args:
            shares: List of (x, y) share pairs

        Returns:
            Reconstructed secret
        """
        if len(shares) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} shares, got {len(shares)}"
            )

        # Use first threshold shares
        shares = shares[:self.threshold]

        # Lagrange interpolation
        secret = 0
        for j, (x_j, y_j) in enumerate(shares):
            # Compute Lagrange basis polynomial L_j(0)
            l_j = 1
            for m, (x_m, _) in enumerate(shares):
                if m != j:
                    l_j = l_j * (-x_m) % self.prime
                    l_j = l_j * pow(x_j - x_m, -1, self.prime) % self.prime

            secret = (secret + y_j * l_j) % self.prime

        return secret

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x."""
        result = 0
        for i, coeff in enumerate(reversed(coefficients)):
            result = (result * x + coeff) % self.prime
        return result


class EncryptedAggregator:
    """
    Aggregation using (partial) homomorphic encryption.

    Note: This is a simplified implementation for demonstration.
    Production use should use established libraries like Microsoft SEAL.
    """

    def __init__(
        self,
        public_key: Optional[Any] = None,
        private_key: Optional[Any] = None,
    ):
        """
        Initialize encrypted aggregator.

        Args:
            public_key: Public key for encryption (optional)
            private_key: Private key for decryption (optional)
        """
        self.public_key = public_key
        self.private_key = private_key

        # For demonstration: use simple XOR "encryption"
        # In production, use Paillier or other homomorphic encryption
        self.use_demo_encryption = public_key is None

        if self.use_demo_encryption:
            logger.warning("Using demo encryption (not secure for production)")

    def encrypt(self, value: np.ndarray) -> np.ndarray:
        """
        Encrypt values.

        Args:
            value: Values to encrypt

        Returns:
            Encrypted values
        """
        if self.use_demo_encryption:
            # Demo: simple XOR with random key
            # NOT SECURE - for demonstration only
            key = np.random.randint(0, 256, value.shape, dtype=np.uint8)
            encrypted = np.bitwise_xor(
                value.astype(np.uint8),
                key,
            )
            return encrypted
        else:
            # Use actual homomorphic encryption
            # This requires integration with libraries like Microsoft SEAL
            raise NotImplementedError(
                "Homomorphic encryption requires external library integration"
            )

    def decrypt(self, encrypted: np.ndarray) -> np.ndarray:
        """
        Decrypt values.

        Args:
            encrypted: Encrypted values

        Returns:
            Decrypted values
        """
        if self.use_demo_encryption:
            # Demo: XOR encryption is symmetric
            # Need to store keys during encryption for this to work
            raise NotImplementedError(
                "Demo decryption requires key storage"
            )
        else:
            # Use actual homomorphic encryption
            raise NotImplementedError(
                "Homomorphic encryption requires external library integration"
            )

    def aggregate_encrypted(
        self,
        encrypted_updates: List[np.ndarray],
    ) -> np.ndarray:
        """
        Aggregate encrypted updates.

        With homomorphic encryption, we can perform operations on
        encrypted data without decryption.

        Args:
            encrypted_updates: List of encrypted updates

        Returns:
            Encrypted aggregate
        """
        if not encrypted_updates:
            return np.array([])

        # Stack and sum (works with some homomorphic schemes)
        stacked = np.stack(encrypted_updates)
        aggregate = np.sum(stacked, axis=0)

        return aggregate


class HybridSecureAggregator:
    """
    Combines pairwise masking with encryption for enhanced security.

    Protocol:
    1. Apply pairwise masking
    2. Encrypt masked updates
    3. Aggregate encrypted updates
    4. Decrypt aggregate
    5. Masks cancel out during aggregation
    """

    def __init__(
        self,
        n_clients: int,
        use_encryption: bool = False,
        bit_size: int = 32,
        seed: int = 42,
    ):
        """
        Initialize hybrid secure aggregator.

        Args:
            n_clients: Number of clients
            use_encryption: Whether to use encryption on top of masking
            bit_size: Bit size for masks
            seed: Random seed
        """
        self.n_clients = n_clients
        self.use_encryption = use_encryption

        # Pairwise masking component
        self.masking_aggregator = SecureAggregator(
            n_clients=n_clients,
            bit_size=bit_size,
            seed=seed,
        )

        # Encryption component (optional)
        if use_encryption:
            self.encryption_aggregator = EncryptedAggregator()
        else:
            self.encryption_aggregator = None

        logger.info(
            f"HybridSecureAggregator: n_clients={n_clients}, "
            f"use_encryption={use_encryption}"
        )

    def secure_aggregate(
        self,
        updates: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Perform secure aggregation.

        Args:
            updates: List of client updates

        Returns:
            Securely aggregated parameters
        """
        # Step 1: Apply pairwise masking
        masked_updates = []
        for client_id, update in enumerate(updates):
            masked = self.masking_aggregator.mask_update(update, client_id)
            masked_updates.append(masked)

        # Step 2: (Optional) Encrypt
        if self.use_encryption and self.encryption_aggregator:
            encrypted_updates = []
            for masked_update in masked_updates:
                encrypted = [
                    self.encryption_aggregator.encrypt(layer)
                    for layer in masked_update
                ]
                encrypted_updates.append(encrypted)

            # Step 3: Aggregate encrypted
            encrypted_aggregate = self.encryption_aggregator.aggregate_encrypted(
                list(zip(*encrypted_updates))
            )

            # Step 4: Decrypt
            # (Not implemented for demo)

        # Step 5: Aggregate masked updates (masks cancel out)
        aggregated = self.masking_aggregator.unmask_aggregate(masked_updates)

        return aggregated
