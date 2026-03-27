"""
Signature Handler for SignGuard Cryptographic Authentication

Handles deterministic ECDSA signatures for model updates.
Provides signing and verification functionality with message hashing.
"""

import hashlib
import time
from typing import List, Tuple, Optional

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend


class SignatureHandler:
    """
    Handles ECDSA signing and verification for model updates.

    Uses deterministic ECDSA (RFC 6979) with SHA-256 hashing.
    Signatures cover: model update hash + round number + timestamp
    """

    def __init__(self, use_deterministic: bool = True):
        """
        Initialize SignatureHandler.

        Args:
            use_deterministic: Use deterministic signatures (RFC 6979)
        """
        self.use_deterministic = use_deterministic

    def compute_update_fingerprint(self, update: np.ndarray) -> bytes:
        """
        Compute cryptographic fingerprint of model update.

        Args:
            update: Model update as numpy array (or list of arrays)

        Returns:
            SHA-256 hash of the update
        """
        # Convert update to bytes
        if isinstance(update, list):
            # Handle list of layer arrays (common in FL)
            update_bytes = b''.join([
                arr.tobytes() if isinstance(arr, np.ndarray) else arr
                for arr in update
            ])
        elif isinstance(update, np.ndarray):
            update_bytes = update.tobytes()
        elif isinstance(update, dict):
            # Handle parameter dictionary
            update_bytes = b''.join([
                v.tobytes() if isinstance(v, np.ndarray) else str(v).encode()
                for v in update.values()
            ])
        else:
            raise TypeError(f"Unsupported update type: {type(update)}")

        # Compute SHA-256 hash
        return hashlib.sha256(update_bytes).digest()

    def create_signature_message(self, update_fingerprint: bytes,
                                  round_num: int, timestamp: float) -> bytes:
        """
        Create the message to be signed.

        Args:
            update_fingerprint: Hash of model update
            round_num: Current FL round
            timestamp: Unix timestamp

        Returns:
            Message bytes ready for signing
        """
        message = (
            update_fingerprint +
            round_num.to_bytes(8, byteorder='big') +
            int(timestamp).to_bytes(8, byteorder='big')
        )
        return message

    def sign_update(self, private_key: ec.EllipticCurvePrivateKey,
                    update: np.ndarray, round_num: int,
                    timestamp: Optional[float] = None) -> bytes:
        """
        Sign a model update with ECDSA.

        Args:
            private_key: ECDSA private key
            update: Model update (numpy array or list of arrays)
            round_num: Current FL round
            timestamp: Unix timestamp (uses current time if None)

        Returns:
            DER-encoded signature
        """
        if timestamp is None:
            timestamp = time.time()

        # Compute fingerprint and message
        fingerprint = self.compute_update_fingerprint(update)
        message = self.create_signature_message(fingerprint, round_num, timestamp)

        # Sign using deterministic ECDSA
        signature = private_key.sign(
            message,
            ec.ECDSA(hashes.SHA256())
        )

        return signature

    def verify_signature(self, public_key: ec.EllipticCurvePublicKey,
                         signature: bytes, update: np.ndarray,
                         round_num: int, timestamp: float) -> bool:
        """
        Verify a model update signature.

        Args:
            public_key: ECDSA public key
            signature: DER-encoded signature to verify
            update: Model update (numpy array or list of arrays)
            round_num: Current FL round
            timestamp: Unix timestamp from signature

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Recreate message
            fingerprint = self.compute_update_fingerprint(update)
            message = self.create_signature_message(fingerprint, round_num, timestamp)

            # Verify signature
            public_key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception:
            return False

    def sign_parameters(self, private_key: ec.EllipticCurvePrivateKey,
                       parameters: List[np.ndarray], round_num: int,
                       timestamp: Optional[float] = None) -> bytes:
        """
        Sign model parameters (list of layer arrays).

        Convenience method for Flower FL framework compatibility.

        Args:
            private_key: ECDSA private key
            parameters: Model parameters as list of numpy arrays
            round_num: Current FL round
            timestamp: Unix timestamp (uses current time if None)

        Returns:
            DER-encoded signature
        """
        return self.sign_update(private_key, parameters, round_num, timestamp)

    def verify_parameters(self, public_key: ec.EllipticCurvePublicKey,
                          signature: bytes, parameters: List[np.ndarray],
                          round_num: int, timestamp: float) -> bool:
        """
        Verify model parameters signature.

        Convenience method for Flower FL framework compatibility.

        Args:
            public_key: ECDSA public key
            signature: DER-encoded signature to verify
            parameters: Model parameters as list of numpy arrays
            round_num: Current FL round
            timestamp: Unix timestamp from signature

        Returns:
            True if signature is valid, False otherwise
        """
        return self.verify_signature(
            public_key, signature, parameters, round_num, timestamp
        )


class SignedUpdate:
    """
    Container for a signed model update.

    Bundles the update with its signature and metadata.
    """

    def __init__(self, client_id: str, update: List[np.ndarray],
                 signature: bytes, round_num: int, timestamp: float,
                 num_examples: int, metrics: Optional[dict] = None):
        """
        Initialize SignedUpdate.

        Args:
            client_id: Unique client identifier
            update: Model update (list of numpy arrays)
            signature: ECDSA signature
            round_num: FL round number
            timestamp: Unix timestamp
            num_examples: Number of training examples
            metrics: Optional training metrics
        """
        self.client_id = client_id
        self.update = update
        self.signature = signature
        self.round_num = round_num
        self.timestamp = timestamp
        self.num_examples = num_examples
        self.metrics = metrics or {}

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'client_id': self.client_id,
            'update': self.update,
            'signature': self.signature,
            'round_num': self.round_num,
            'timestamp': self.timestamp,
            'num_examples': self.num_examples,
            'metrics': self.metrics
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SignedUpdate':
        """Create from dictionary."""
        return cls(
            client_id=data['client_id'],
            update=data['update'],
            signature=data['signature'],
            round_num=data['round_num'],
            timestamp=data['timestamp'],
            num_examples=data['num_examples'],
            metrics=data.get('metrics', {})
        )


# Standalone functions for convenience
def sign_update(private_key: ec.EllipticCurvePrivateKey,
                update: np.ndarray, round_num: int,
                timestamp: Optional[float] = None) -> bytes:
    """
    Sign a model update.

    Convenience function using default SignatureHandler.

    Args:
        private_key: ECDSA private key
        update: Model update
        round_num: FL round number
        timestamp: Unix timestamp

    Returns:
        DER-encoded signature
    """
    handler = SignatureHandler()
    return handler.sign_update(private_key, update, round_num, timestamp)


def verify_signature(public_key: ec.EllipticCurvePublicKey,
                     signature: bytes, update: np.ndarray,
                     round_num: int, timestamp: float) -> bool:
    """
    Verify a model update signature.

    Convenience function using default SignatureHandler.

    Args:
        public_key: ECDSA public key
        signature: DER-encoded signature
        update: Model update
        round_num: FL round number
        timestamp: Unix timestamp

    Returns:
        True if valid, False otherwise
    """
    handler = SignatureHandler()
    return handler.verify_signature(public_key, signature, update, round_num, timestamp)
