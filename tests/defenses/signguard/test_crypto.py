"""
Unit Tests for SignGuard Cryptographic Layer

Tests key generation, signing, verification, and batch verification.
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

from src.crypto.key_manager import KeyManager, generate_key_pair
from src.crypto.signature_handler import SignatureHandler, SignedUpdate
from src.crypto.batch_verifier import BatchVerifier


class TestKeyManager:
    """Test KeyManager functionality."""

    def test_generate_key_pair(self):
        """Test key pair generation."""
        km = KeyManager()
        private_pem, public_pem = km.generate_key_pair("client_1")

        # Check PEM format
        assert private_pem.startswith(b'-----BEGIN PRIVATE KEY-----')
        assert public_pem.startswith(b'-----BEGIN PUBLIC KEY-----')

        # Check keys are stored
        assert km.client_exists("client_1")
        assert km.get_private_key("client_1") is not None
        assert km.get_public_key("client_1") is not None

    def test_duplicate_client_error(self):
        """Test error when generating key for existing client."""
        km = KeyManager()
        km.generate_key_pair("client_1")

        with pytest.raises(ValueError, match="already has a registered key"):
            km.generate_key_pair("client_1")

    def test_key_persistence(self):
        """Test key persistence to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            km = KeyManager(key_dir=tmpdir)
            private_pem, public_pem = km.generate_key_pair("client_1")

            # Create new KeyManager and load keys
            km2 = KeyManager(key_dir=tmpdir)

            # Check keys were loaded
            assert km2.client_exists("client_1")
            assert km2.get_public_key_pem("client_1") == public_pem

    def test_register_client(self):
        """Test registering a client with existing public key."""
        km = KeyManager()
        _, public_pem = km.generate_key_pair("client_1")

        # Register new client with client_1's public key
        km.register_client("client_2", public_pem)

        # Both clients should exist
        assert km.client_exists("client_2")

        # Public key should match
        assert km.get_public_key_pem("client_2") == public_pem

    def test_remove_client(self):
        """Test removing a client."""
        with tempfile.TemporaryDirectory() as tmpdir:
            km = KeyManager(key_dir=tmpdir)
            km.generate_key_pair("client_1")

            # Remove client
            km.remove_client("client_1")

            # Check removed
            assert not km.client_exists("client_1")

            # Check files deleted
            assert not (Path(tmpdir) / "client_1_private.pem").exists()
            assert not (Path(tmpdir) / "client_1_public.pem").exists()

    def test_list_clients(self):
        """Test listing all clients."""
        km = KeyManager()
        km.generate_key_pair("client_1")
        km.generate_key_pair("client_2")
        km.generate_key_pair("client_3")

        clients = km.list_clients()
        assert set(clients) == {"client_1", "client_2", "client_3"}

    def test_standalone_generate_key_pair(self):
        """Test standalone key generation function."""
        private_pem, public_pem = generate_key_pair()

        assert private_pem.startswith(b'-----BEGIN PRIVATE KEY-----')
        assert public_pem.startswith(b'-----BEGIN PUBLIC KEY-----')


class TestSignatureHandler:
    """Test SignatureHandler functionality."""

    def test_compute_update_fingerprint(self):
        """Test fingerprint computation."""
        handler = SignatureHandler()
        update = np.random.randn(100, 100)

        fingerprint1 = handler.compute_update_fingerprint(update)
        fingerprint2 = handler.compute_update_fingerprint(update)

        # Same update should produce same fingerprint
        assert fingerprint1 == fingerprint2

        # Different update should produce different fingerprint
        different_update = np.random.randn(100, 100)
        fingerprint3 = handler.compute_update_fingerprint(different_update)
        assert fingerprint1 != fingerprint3

    def test_sign_and_verify_update(self):
        """Test signing and verification of updates."""
        handler = SignatureHandler()

        # Generate keys
        private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
        public_key = private_key.public_key()

        # Create update
        update = np.random.randn(10, 10).astype(np.float32)
        round_num = 5
        timestamp = time.time()

        # Sign
        signature = handler.sign_update(private_key, update, round_num, timestamp)

        # Verify
        assert handler.verify_signature(public_key, signature, update, round_num, timestamp)

    def test_invalid_signature(self):
        """Test verification fails for invalid signature."""
        handler = SignatureHandler()

        # Generate keys
        private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
        public_key = private_key.public_key()

        # Create update
        update = np.random.randn(10, 10).astype(np.float32)
        round_num = 5
        timestamp = time.time()

        # Sign
        signature = handler.sign_update(private_key, update, round_num, timestamp)

        # Tamper with signature
        tampered_signature = signature[:-5] + b'\x00\x00\x00\x00\x00'

        # Verify should fail
        assert not handler.verify_signature(public_key, tampered_signature, update, round_num, timestamp)

    def test_sign_and_verify_parameters(self):
        """Test signing and verification of parameter lists (FL format)."""
        handler = SignatureHandler()

        # Generate keys
        private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
        public_key = private_key.public_key()

        # Create parameter list
        parameters = [
            np.random.randn(100, 50).astype(np.float32),
            np.random.randn(50).astype(np.float32),
            np.random.randn(50, 10).astype(np.float32)
        ]
        round_num = 10
        timestamp = time.time()

        # Sign
        signature = handler.sign_parameters(private_key, parameters, round_num, timestamp)

        # Verify
        assert handler.verify_parameters(public_key, signature, parameters, round_num, timestamp)

    def test_signed_update_container(self):
        """Test SignedUpdate container class."""
        handler = SignatureHandler()

        # Generate keys
        private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())

        # Create update
        update = [np.random.randn(100).astype(np.float32)]
        round_num = 5
        timestamp = time.time()
        signature = handler.sign_parameters(private_key, update, round_num, timestamp)

        # Create SignedUpdate
        signed_update = SignedUpdate(
            client_id="client_1",
            update=update,
            signature=signature,
            round_num=round_num,
            timestamp=timestamp,
            num_examples=100,
            metrics={"loss": 0.5}
        )

        # Check attributes
        assert signed_update.client_id == "client_1"
        assert signed_update.round_num == 5
        assert signed_update.num_examples == 100
        assert signed_update.metrics["loss"] == 0.5

        # Test serialization
        data = signed_update.to_dict()
        restored = SignedUpdate.from_dict(data)

        assert restored.client_id == signed_update.client_id
        assert restored.round_num == signed_update.round_num


class TestBatchVerifier:
    """Test BatchVerifier functionality."""

    def test_verify_batch(self):
        """Test batch verification of multiple signatures."""
        verifier = BatchVerifier()

        # Generate keys and create updates
        num_updates = 5
        public_keys = []
        signed_updates = []

        for i in range(num_updates):
            # Generate keys
            private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
            public_key = private_key.public_key()
            public_keys.append(public_key)

            # Create update
            update = [np.random.randn(50).astype(np.float32)]
            round_num = 5
            timestamp = time.time()

            # Sign
            handler = SignatureHandler()
            signature = handler.sign_parameters(private_key, update, round_num, timestamp)

            # Create SignedUpdate
            signed_update = SignedUpdate(
                client_id=f"client_{i}",
                update=update,
                signature=signature,
                round_num=round_num,
                timestamp=timestamp,
                num_examples=100
            )
            signed_updates.append(signed_update)

        # Verify batch
        pairs = list(zip(public_keys, signed_updates))
        results = verifier.verify_batch(pairs)

        # All should be valid
        assert len(results) == num_updates
        assert all(results)

    def test_verify_batch_with_invalid_signature(self):
        """Test batch verification with one invalid signature."""
        verifier = BatchVerifier()

        # Generate keys and create updates
        num_updates = 5
        public_keys = []
        signed_updates = []

        for i in range(num_updates):
            private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
            public_key = private_key.public_key()
            public_keys.append(public_key)

            update = [np.random.randn(50).astype(np.float32)]
            round_num = 5
            timestamp = time.time()

            handler = SignatureHandler()
            signature = handler.sign_parameters(private_key, update, round_num, timestamp)

            # Tamper with one signature
            if i == 2:
                signature = signature[:-5] + b'\x00\x00\x00\x00\x00'

            signed_update = SignedUpdate(
                client_id=f"client_{i}",
                update=update,
                signature=signature,
                round_num=round_num,
                timestamp=timestamp,
                num_examples=100
            )
            signed_updates.append(signed_update)

        # Verify batch
        pairs = list(zip(public_keys, signed_updates))
        results = verifier.verify_batch(pairs)

        # Check results
        assert len(results) == num_updates
        assert results[0]  # Valid
        assert results[1]  # Valid
        assert not results[2]  # Invalid (tampered)
        assert results[3]  # Valid
        assert results[4]  # Valid

    def test_filter_valid_updates(self):
        """Test filtering to keep only valid updates."""
        verifier = BatchVerifier()

        # Create updates with one invalid
        updates_dict = {}

        for i in range(3):
            private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
            public_key = private_key.public_key()

            update = [np.random.randn(50).astype(np.float32)]
            round_num = 5
            timestamp = time.time()

            handler = SignatureHandler()
            signature = handler.sign_parameters(private_key, update, round_num, timestamp)

            # Tamper with client_1's signature
            if i == 1:
                signature = signature[:-5] + b'\x00\x00\x00\x00\x00'

            signed_update = SignedUpdate(
                client_id=f"client_{i}",
                update=update,
                signature=signature,
                round_num=round_num,
                timestamp=timestamp,
                num_examples=100
            )

            updates_dict[f"client_{i}"] = (public_key, signed_update)

        # Filter valid updates
        valid_updates = verifier.filter_valid_updates(updates_dict)

        # Should have 2 valid updates
        assert len(valid_updates) == 2
        assert "client_0" in valid_updates
        assert "client_1" not in valid_updates  # Invalid
        assert "client_2" in valid_updates

    def test_get_verification_stats(self):
        """Test verification statistics computation."""
        verifier = BatchVerifier()
        results = [True, True, False, True, False]

        stats = verifier.get_verification_stats(results)

        assert stats['total'] == 5
        assert stats['valid'] == 3
        assert stats['invalid'] == 2
        assert stats['validity_rate'] == 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
