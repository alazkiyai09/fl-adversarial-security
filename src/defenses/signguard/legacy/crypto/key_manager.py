"""
Key Manager for SignGuard Cryptographic Authentication

Handles ECDSA key generation, storage, and retrieval using secp256k1 curve.
Provides deterministic key management for federated learning clients.
"""

import os
import pickle
from typing import Dict, Optional, Tuple
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


class KeyManager:
    """
    Manages ECDSA keys for federated learning clients.

    Uses secp256k1 curve (same as Bitcoin) for efficient digital signatures.
    Supports key generation, storage, and retrieval.
    """

    def __init__(self, key_dir: Optional[str] = None):
        """
        Initialize KeyManager.

        Args:
            key_dir: Directory to store keys. If None, keys are in-memory only.
        """
        self.key_dir = Path(key_dir) if key_dir else None
        self.keys: Dict[str, Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]] = {}

        if self.key_dir:
            self.key_dir.mkdir(parents=True, exist_ok=True)
            self._load_existing_keys()

    def generate_key_pair(self, client_id: str) -> Tuple[bytes, bytes]:
        """
        Generate a new ECDSA key pair for a client.

        Args:
            client_id: Unique identifier for the client

        Returns:
            Tuple of (private_key_pem, public_key_pem) in PEM format

        Raises:
            ValueError: If client_id already has a key
        """
        if client_id in self.keys:
            raise ValueError(f"Client {client_id} already has a registered key")

        # Generate private key using secp256k1 curve
        private_key = ec.generate_private_key(
            ec.SECP256K1(),
            backend=default_backend()
        )

        # Derive public key
        public_key = private_key.public_key()

        # Store in memory
        self.keys[client_id] = (private_key, public_key)

        # Serialize to PEM format
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Save to disk if key_dir is set
        if self.key_dir:
            self._save_key_pair(client_id, private_key, public_key)

        return private_pem, public_pem

    def get_private_key(self, client_id: str) -> Optional[ec.EllipticCurvePrivateKey]:
        """
        Retrieve private key for a client.

        Args:
            client_id: Client identifier

        Returns:
            Private key object, or None if not found
        """
        if client_id not in self.keys:
            return None
        return self.keys[client_id][0]

    def get_public_key(self, client_id: str) -> Optional[ec.EllipticCurvePublicKey]:
        """
        Retrieve public key for a client.

        Args:
            client_id: Client identifier

        Returns:
            Public key object, or None if not found
        """
        if client_id not in self.keys:
            return None
        return self.keys[client_id][1]

    def get_public_key_pem(self, client_id: str) -> Optional[bytes]:
        """
        Retrieve public key in PEM format for a client.

        Args:
            client_id: Client identifier

        Returns:
            Public key in PEM format, or None if not found
        """
        public_key = self.get_public_key(client_id)
        if public_key is None:
            return None

        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def register_client(self, client_id: str, public_key_pem: bytes) -> None:
        """
        Register a client with an existing public key.

        Used when server receives a client's public key during registration.

        Args:
            client_id: Client identifier
            public_key_pem: Public key in PEM format

        Raises:
            ValueError: If client_id already registered
        """
        if client_id in self.keys:
            raise ValueError(f"Client {client_id} already registered")

        # Load public key from PEM
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )

        # Store public key only (no private key on server)
        self.keys[client_id] = (None, public_key)

    def client_exists(self, client_id: str) -> bool:
        """
        Check if a client is registered.

        Args:
            client_id: Client identifier

        Returns:
            True if client exists, False otherwise
        """
        return client_id in self.keys

    def remove_client(self, client_id: str) -> None:
        """
        Remove a client's keys from memory.

        Args:
            client_id: Client identifier
        """
        if client_id in self.keys:
            del self.keys[client_id]

        # Also remove from disk if key_dir is set
        if self.key_dir:
            private_key_path = self.key_dir / f"{client_id}_private.pem"
            public_key_path = self.key_dir / f"{client_id}_public.pem"
            private_key_path.unlink(missing_ok=True)
            public_key_path.unlink(missing_ok=True)

    def list_clients(self) -> list:
        """
        List all registered client IDs.

        Returns:
            List of client IDs
        """
        return list(self.keys.keys())

    def _save_key_pair(self, client_id: str,
                       private_key: ec.EllipticCurvePrivateKey,
                       public_key: ec.EllipticCurvePublicKey) -> None:
        """Save key pair to disk."""
        # Save private key
        private_key_path = self.key_dir / f"{client_id}_private.pem"
        with open(private_key_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Save public key
        public_key_path = self.key_dir / f"{client_id}_public.pem"
        with open(public_key_path, 'wb') as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))

    def _load_existing_keys(self) -> None:
        """Load existing keys from disk."""
        if not self.key_dir:
            return

        # Load all public keys
        for public_key_path in self.key_dir.glob("*_public.pem"):
            client_id = public_key_path.stem.replace("_public", "")

            # Load public key
            with open(public_key_path, 'rb') as f:
                public_key_pem = f.read()
                public_key = serialization.load_pem_public_key(
                    public_key_pem,
                    backend=default_backend()
                )

            # Try to load private key (may not exist on server)
            private_key_path = self.key_dir / f"{client_id}_private.pem"
            private_key = None
            if private_key_path.exists():
                with open(private_key_path, 'rb') as f:
                    private_key_pem = f.read()
                    private_key = serialization.load_pem_private_key(
                        private_key_pem,
                        password=None,
                        backend=default_backend()
                    )

            self.keys[client_id] = (private_key, public_key)


def generate_key_pair() -> Tuple[bytes, bytes]:
    """
    Standalone function to generate a key pair without persistence.

    Returns:
        Tuple of (private_key_pem, public_key_pem)
    """
    private_key = ec.generate_private_key(
        ec.SECP256K1(),
        backend=default_backend()
    )
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    return private_pem, public_pem
