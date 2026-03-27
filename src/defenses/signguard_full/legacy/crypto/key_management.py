"""Key management for SignGuard cryptographic operations."""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

from src.defenses.signguard_full.legacy.crypto.signature import SignatureManager


class KeyManager:
    """Manages generation, storage, and loading of cryptographic keys."""

    def __init__(
        self,
        keys_dir: Union[Path, str] = "keys",
        signature_manager: Optional[SignatureManager] = None,
    ):
        """Initialize key manager.

        Args:
            keys_dir: Directory to store keys
            signature_manager: Optional signature manager (creates default if None)
        """
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        self.signature_manager = signature_manager or SignatureManager()

    def generate_and_save_keys(
        self,
        client_id: str,
        password: Optional[bytes] = None,
        require_encryption: bool = True,
    ) -> Tuple[str, ...]:
        """Generate and save key pair for a client.

        Args:
            client_id: Client identifier
            password: Optional password to encrypt private key (required if require_encryption=True)
            require_encryption: Whether to require password for private key storage

        Returns:
            Tuple of (private_key_path, public_key_path)

        Raises:
            ValueError: If require_encryption=True but no password provided
        """
        if require_encryption and password is None:
            raise ValueError(
                "Password must be provided when require_encryption=True. "
                "For production use, always encrypt private keys at rest."
            )

        if password is None and not require_encryption:
            import warnings
            warnings.warn(
                "Saving private key without encryption. This is not recommended for production.",
                SecurityWarning,
                stacklevel=2
            )

        # Generate key pair
        private_key, public_key = self.signature_manager.generate_keypair()

        # Serialize keys
        private_key_str = self.signature_manager.serialize_private_key(
            private_key, password
        )
        public_key_str = self.signature_manager.serialize_public_key(public_key)

        # Save to files
        private_key_path = self.keys_dir / f"{client_id}_private.pem"
        public_key_path = self.keys_dir / f"{client_id}_public.pem"

        private_key_path.write_text(private_key_str)
        public_key_path.write_text(public_key_str)

        # Save metadata
        self._save_key_metadata(client_id, public_key)

        return str(private_key_path), str(public_key_path)

    def load_private_key(
        self,
        client_id: str,
        password: Optional[bytes] = None,
    ) -> ec.EllipticCurvePrivateKey:
        """Load private key for a client.

        Args:
            client_id: Client identifier
            password: Optional password for decryption

        Returns:
            Private key object

        Raises:
            FileNotFoundError: If key file doesn't exist
        """
        private_key_path = self.keys_dir / f"{client_id}_private.pem"
        
        if not private_key_path.exists():
            raise FileNotFoundError(f"Private key not found for client: {client_id}")

        private_key_str = private_key_path.read_text()
        return self.signature_manager.deserialize_private_key(
            private_key_str, password
        )

    def load_public_key(self, client_id: str) -> ec.EllipticCurvePublicKey:
        """Load public key for a client.

        Args:
            client_id: Client identifier

        Returns:
            Public key object

        Raises:
            FileNotFoundError: If key file doesn't exist
        """
        public_key_path = self.keys_dir / f"{client_id}_public.pem"
        
        if not public_key_path.exists():
            raise FileNotFoundError(f"Public key not found for client: {client_id}")

        public_key_str = public_key_path.read_text()
        return self.signature_manager.deserialize_public_key(public_key_str)

    def load_public_key_string(self, client_id: str) -> str:
        """Load public key as base64 string.

        Args:
            client_id: Client identifier

        Returns:
            Base64-encoded public key string

        Raises:
            FileNotFoundError: If key file doesn't exist
        """
        public_key_path = self.keys_dir / f"{client_id}_public.pem"
        
        if not public_key_path.exists():
            raise FileNotFoundError(f"Public key not found for client: {client_id}")

        return public_key_path.read_text()

    def client_has_keys(self, client_id: str) -> bool:
        """Check if keys exist for a client.

        Args:
            client_id: Client identifier

        Returns:
            True if both private and public keys exist
        """
        private_key_path = self.keys_dir / f"{client_id}_private.pem"
        public_key_path = self.keys_dir / f"{client_id}_public.pem"
        
        return private_key_path.exists() and public_key_path.exists()

    def delete_keys(self, client_id: str) -> None:
        """Delete keys for a client.

        Args:
            client_id: Client identifier
        """
        private_key_path = self.keys_dir / f"{client_id}_private.pem"
        public_key_path = self.keys_dir / f"{client_id}_public.pem"
        metadata_path = self.keys_dir / f"{client_id}_metadata.json"

        if private_key_path.exists():
            private_key_path.unlink()
        if public_key_path.exists():
            public_key_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

    def list_clients(self) -> List[str]:
        """List all clients with stored keys.

        Returns:
            List of client IDs
        """
        clients = set()
        for file in self.keys_dir.glob("*_public.pem"):
            client_id = file.stem.replace("_public", "")
            clients.add(client_id)
        return sorted(clients)

    def _save_key_metadata(self, client_id: str, public_key: ec.EllipticCurvePublicKey) -> None:
        """Save metadata about a client's public key.

        Args:
            client_id: Client identifier
            public_key: Public key object
        """
        metadata_path = self.keys_dir / f"{client_id}_metadata.json"
        
        info = self.signature_manager.get_key_info(public_key)
        info["client_id"] = client_id
        
        metadata_path.write_text(json.dumps(info, indent=2))

    def rotate_keys(
        self,
        client_id: str,
        password: Optional[bytes] = None,
        backup_old: bool = True,
    ) -> Tuple[str, ...]:
        """Generate new keys for a client (key rotation).

        Args:
            client_id: Client identifier
            password: Optional password to encrypt new private key
            backup_old: Whether to backup old keys

        Returns:
            Tuple of (new_private_key_path, new_public_key_path)
        """
        # Backup old keys if requested
        if backup_old and self.client_has_keys(client_id):
            old_private = self.keys_dir / f"{client_id}_private.pem"
            old_public = self.keys_dir / f"{client_id}_public.pem"
            old_metadata = self.keys_dir / f"{client_id}_metadata.json"
            
            timestamp = int(os.times()[4])  # Use process time as simple timestamp
            backup_suffix = f".bak_{timestamp}"
            
            if old_private.exists():
                old_private.rename(str(old_private) + backup_suffix)
            if old_public.exists():
                old_public.rename(str(old_public) + backup_suffix)
            if old_metadata.exists():
                old_metadata.rename(str(old_metadata) + backup_suffix)

        # Generate new keys
        return self.generate_and_save_keys(client_id, password)


class KeyStore:
    """In-memory key store for testing and simulation."""

    def __init__(self, signature_manager: Optional[SignatureManager] = None):
        """Initialize in-memory key store.

        Args:
            signature_manager: Optional signature manager
        """
        self.signature_manager = signature_manager or SignatureManager()
        self._private_keys: Dict[str, ec.EllipticCurvePrivateKey] = {}
        self._public_keys: Dict[str, ec.EllipticCurvePublicKey] = {}

    def generate_keypair(self, client_id: str) -> None:
        """Generate and store key pair in memory.

        Args:
            client_id: Client identifier
        """
        private_key, public_key = self.signature_manager.generate_keypair()
        self._private_keys[client_id] = private_key
        self._public_keys[client_id] = public_key

    def get_private_key(self, client_id: str) -> ec.EllipticCurvePrivateKey:
        """Get private key for client.

        Args:
            client_id: Client identifier

        Returns:
            Private key object

        Raises:
            KeyError: If key doesn't exist
        """
        if client_id not in self._private_keys:
            raise KeyError(f"No private key for client: {client_id}")
        return self._private_keys[client_id]

    def get_public_key(self, client_id: str) -> ec.EllipticCurvePublicKey:
        """Get public key for client.

        Args:
            client_id: Client identifier

        Returns:
            Public key object

        Raises:
            KeyError: If key doesn't exist
        """
        if client_id not in self._public_keys:
            raise KeyError(f"No public key for client: {client_id}")
        return self._public_keys[client_id]

    def get_public_key_string(self, client_id: str) -> str:
        """Get public key as base64 string.

        Args:
            client_id: Client identifier

        Returns:
            Base64-encoded public key string

        Raises:
            KeyError: If key doesn't exist
        """
        public_key = self.get_public_key(client_id)
        return self.signature_manager.serialize_public_key(public_key)

    def has_client(self, client_id: str) -> bool:
        """Check if client exists in store.

        Args:
            client_id: Client identifier

        Returns:
            True if client has keys
        """
        return client_id in self._private_keys

    def remove_client(self, client_id: str) -> None:
        """Remove client keys from store.

        Args:
            client_id: Client identifier
        """
        self._private_keys.pop(client_id, None)
        self._public_keys.pop(client_id, None)

    def list_clients(self) -> List[str]:
        """List all clients in store.

        Returns:
            List of client IDs
        """
        return list(self._private_keys.keys())
