"""
Batch Verifier for SignGuard Cryptographic Authentication

Provides efficient batch verification of multiple signatures.
Reduces verification overhead for federated learning aggregation.
"""

import time
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ec

from .signature_handler import SignatureHandler, SignedUpdate


class BatchVerifier:
    """
    Verifies multiple signatures in parallel with early termination.

    Features:
    - Parallel verification using ThreadPoolExecutor
    - Early termination on first invalid signature
    - Detailed verification reports
    """

    def __init__(self, signature_handler: Optional[SignatureHandler] = None,
                 max_workers: int = 4):
        """
        Initialize BatchVerifier.

        Args:
            signature_handler: SignatureHandler instance (creates default if None)
            max_workers: Maximum number of parallel verification threads
        """
        self.signature_handler = signature_handler or SignatureHandler()
        self.max_workers = max_workers

    def verify_batch(self, signed_updates: List[Tuple[ec.EllipticCurvePublicKey,
                                                       SignedUpdate]]) -> List[bool]:
        """
        Verify a batch of signed updates.

        Args:
            signed_updates: List of (public_key, signed_update) tuples

        Returns:
            List of booleans indicating validity (same order as input)
        """
        results = [None] * len(signed_updates)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all verification tasks
            future_to_index = {
                executor.submit(
                    self._verify_single,
                    public_key,
                    signed_update
                ): idx
                for idx, (public_key, signed_update) in enumerate(signed_updates)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Verification error for index {idx}: {e}")
                    results[idx] = False

        return results

    def verify_batch_with_early_stop(self,
                                      signed_updates: List[Tuple[ec.EllipticCurvePublicKey,
                                                                 SignedUpdate]],
                                      stop_on_first_invalid: bool = False) -> Tuple[List[bool], int]:
        """
        Verify batch with optional early termination.

        Args:
            signed_updates: List of (public_key, signed_update) tuples
            stop_on_first_invalid: Stop verification if invalid signature found

        Returns:
            Tuple of (results list, num_verified)
        """
        results = [False] * len(signed_updates)
        num_verified = 0

        for idx, (public_key, signed_update) in enumerate(signed_updates):
            is_valid = self._verify_single(public_key, signed_update)
            results[idx] = is_valid
            num_verified += 1

            if stop_on_first_invalid and not is_valid:
                # Fill remaining as unverified
                results[idx+1:] = [False] * (len(results) - idx - 1)
                break

        return results, num_verified

    def verify_batch_dict(self,
                          signed_updates: Dict[str,
                                                Tuple[ec.EllipticCurvePublicKey,
                                                      SignedUpdate]]) -> Dict[str, bool]:
        """
        Verify a batch of updates indexed by client ID.

        Args:
            signed_updates: Dictionary mapping client_id to (public_key, signed_update)

        Returns:
            Dictionary mapping client_id to verification result
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_client = {
                executor.submit(
                    self._verify_single,
                    public_key,
                    signed_update
                ): client_id
                for client_id, (public_key, signed_update) in signed_updates.items()
            }

            # Collect results
            for future in as_completed(future_to_client):
                client_id = future_to_client[future]
                try:
                    results[client_id] = future.result()
                except Exception as e:
                    print(f"Verification error for {client_id}: {e}")
                    results[client_id] = False

        return results

    def filter_valid_updates(self,
                             signed_updates: Dict[str,
                                                   Tuple[ec.EllipticCurvePublicKey,
                                                         SignedUpdate]]) -> Dict[str, SignedUpdate]:
        """
        Filter batch to keep only validly signed updates.

        Args:
            signed_updates: Dictionary mapping client_id to (public_key, signed_update)

        Returns:
            Dictionary of client_id -> signed_update for valid signatures only
        """
        verification_results = self.verify_batch_dict(signed_updates)

        valid_updates = {
            client_id: signed_update
            for client_id, (public_key, signed_update) in signed_updates.items()
            if verification_results.get(client_id, False)
        }

        return valid_updates

    def get_verification_stats(self, results: List[bool]) -> Dict[str, int]:
        """
        Compute verification statistics.

        Args:
            results: List of verification results

        Returns:
            Dictionary with stats: total, valid, invalid, validity_rate
        """
        total = len(results)
        valid = sum(results)
        invalid = total - valid

        return {
            'total': total,
            'valid': valid,
            'invalid': invalid,
            'validity_rate': valid / total if total > 0 else 0.0
        }

    def _verify_single(self,
                       public_key: ec.EllipticCurvePublicKey,
                       signed_update: SignedUpdate) -> bool:
        """
        Verify a single signed update.

        Args:
            public_key: Client's public key
            signed_update: SignedUpdate object

        Returns:
            True if valid, False otherwise
        """
        return self.signature_handler.verify_parameters(
            public_key=public_key,
            signature=signed_update.signature,
            parameters=signed_update.update,
            round_num=signed_update.round_num,
            timestamp=signed_update.timestamp
        )


def verify_signatures_batch(public_keys: List[bytes],
                             signatures: List[bytes],
                             updates: List[List[np.ndarray]],
                             round_nums: List[int],
                             timestamps: List[float],
                             max_workers: int = 4) -> List[bool]:
    """
    Standalone function for batch verification.

    Convenience function using default BatchVerifier.

    Args:
        public_keys: List of PEM-encoded public keys
        signatures: List of DER-encoded signatures
        updates: List of model updates
        round_nums: List of round numbers
        timestamps: List of timestamps
        max_workers: Maximum number of parallel threads

    Returns:
        List of verification results
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend

    # Load public keys from PEM
    public_key_objects = [
        serialization.load_pem_public_key(pk, backend=default_backend())
        for pk in public_keys
    ]

    # Create SignedUpdate objects
    signed_updates = [
        SignedUpdate(
            client_id=f"client_{i}",
            update=updates[i],
            signature=signatures[i],
            round_num=round_nums[i],
            timestamp=timestamps[i],
            num_examples=0  # Not used for verification
        )
        for i in range(len(updates))
    ]

    # Create pairs
    pairs = list(zip(public_key_objects, signed_updates))

    # Verify batch
    verifier = BatchVerifier(max_workers=max_workers)
    return verifier.verify_batch(pairs)
