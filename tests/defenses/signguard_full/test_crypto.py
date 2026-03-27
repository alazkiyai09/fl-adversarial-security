"""Tests for cryptographic modules."""

import pytest
import torch
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import ec

from signguard.crypto import SignatureManager, KeyManager, KeyStore
from signguard.core.types import ModelUpdate, SignedUpdate


class TestSignatureManager:
    """Tests for SignatureManager."""

    @pytest.fixture
    def signature_manager(self):
        """Create signature manager instance."""
        return SignatureManager()

    @pytest.fixture
    def keypair(self, signature_manager):
        """Generate key pair for testing."""
        return signature_manager.generate_keypair()

    @pytest.fixture
    def sample_update(self, seed_randomness, sample_model_params):
        """Create sample model update."""
        return ModelUpdate(
            client_id="test_client",
            round_num=1,
            parameters=sample_model_params,
            num_samples=100,
            metrics={"loss": 0.5, "accuracy": 0.9},
        )

    def test_generate_keypair(self, signature_manager):
        """Test key pair generation."""
        private_key, public_key = signature_manager.generate_keypair()

        assert isinstance(private_key, ec.EllipticCurvePrivateKey)
        assert isinstance(public_key, ec.EllipticCurvePublicKey)
        # Check that public key matches private key
        derived_public = private_key.public_key()
        assert signature_manager.get_key_info(public_key) == signature_manager.get_key_info(derived_public)

    def test_sign_update(self, signature_manager, keypair, sample_update):
        """Test signing a model update."""
        private_key, _ = keypair

        signature = signature_manager.sign_update(sample_update, private_key)

        assert isinstance(signature, str)
        assert len(signature) > 0

    def test_verify_valid_signature(self, signature_manager, keypair, sample_update):
        """Test verifying a valid signature."""
        private_key, public_key = keypair

        # Sign update
        signature = signature_manager.sign_update(sample_update, private_key)

        # Serialize public key
        public_key_str = signature_manager.serialize_public_key(public_key)

        # Create signed update
        signed_update = SignedUpdate(
            update=sample_update,
            signature=signature,
            public_key=public_key_str,
            algorithm="ECDSA",
        )

        # Verify
        assert signature_manager.verify_update(signed_update) is True

    def test_verify_invalid_signature(self, signature_manager, keypair, sample_update):
        """Test verifying an invalid signature."""
        private_key, public_key = keypair

        # Sign update
        signature = signature_manager.sign_update(sample_update, private_key)

        # Tamper with signature
        tampered_signature = "X" + signature[1:]

        # Serialize public key
        public_key_str = signature_manager.serialize_public_key(public_key)

        # Create signed update with tampered signature
        signed_update = SignedUpdate(
            update=sample_update,
            signature=tampered_signature,
            public_key=public_key_str,
            algorithm="ECDSA",
        )

        # Verify should fail
        assert signature_manager.verify_update(signed_update) is False

    def test_verify_tampered_update(self, signature_manager, keypair, sample_update):
        """Test verifying signature on tampered update."""
        private_key, public_key = keypair

        # Sign update
        signature = signature_manager.sign_update(sample_update, private_key)

        # Serialize public key
        public_key_str = signature_manager.serialize_public_key(public_key)

        # Tamper with update
        tampered_update = ModelUpdate(
            client_id=sample_update.client_id,
            round_num=sample_update.round_num,
            parameters=sample_update.parameters,
            num_samples=999,  # Tampered value
            metrics=sample_update.metrics,
        )

        # Create signed update with tampered data
        signed_update = SignedUpdate(
            update=tampered_update,
            signature=signature,
            public_key=public_key_str,
            algorithm="ECDSA",
        )

        # Verify should fail
        assert signature_manager.verify_update(signed_update) is False

    def test_serialize_deserialize_public_key(self, signature_manager, keypair):
        """Test public key serialization/deserialization."""
        _, public_key = keypair

        # Serialize
        key_str = signature_manager.serialize_public_key(public_key)
        assert isinstance(key_str, str)
        assert len(key_str) > 0

        # Deserialize
        restored_key = signature_manager.deserialize_public_key(key_str)

        assert isinstance(restored_key, ec.EllipticCurvePublicKey)

        # Check keys match
        assert signature_manager.get_key_info(public_key) == signature_manager.get_key_info(restored_key)

    def test_serialize_deserialize_private_key(self, signature_manager, keypair):
        """Test private key serialization/deserialization."""
        private_key, _ = keypair

        # Serialize without password
        key_str = signature_manager.serialize_private_key(private_key)
        assert isinstance(key_str, str)
        assert len(key_str) > 0

        # Deserialize
        restored_key = signature_manager.deserialize_private_key(key_str)

        assert isinstance(restored_key, ec.EllipticCurvePrivateKey)

    def test_serialize_private_key_with_password(self, signature_manager, keypair):
        """Test private key serialization with password."""
        private_key, _ = keypair
        password = b"test_password"

        # Serialize with password
        key_str = signature_manager.serialize_private_key(private_key, password)

        # Deserialize with correct password
        restored_key = signature_manager.deserialize_private_key(key_str, password)
        assert isinstance(restored_key, ec.EllipticCurvePrivateKey)

        # Deserialize with wrong password should fail
        with pytest.raises(ValueError):
            signature_manager.deserialize_private_key(key_str, b"wrong_password")

    def test_get_key_info(self, signature_manager, keypair):
        """Test getting key information."""
        _, public_key = keypair

        info = signature_manager.get_key_info(public_key)

        assert "curve" in info
        assert "key_size_bits" in info
        assert "x" in info
        assert "y" in info
        assert info["curve"] == "secp256r1"
        assert info["key_size_bits"] == 256


class TestKeyStore:
    """Tests for in-memory KeyStore."""

    @pytest.fixture
    def key_store(self):
        """Create key store instance."""
        return KeyStore()

    def test_generate_keypair(self, key_store):
        """Test generating key pair."""
        key_store.generate_keypair("client_0")

        assert key_store.has_client("client_0")
        assert "client_0" in key_store.list_clients()

    def test_get_private_key(self, key_store):
        """Test getting private key."""
        key_store.generate_keypair("client_0")

        private_key = key_store.get_private_key("client_0")
        assert isinstance(private_key, ec.EllipticCurvePrivateKey)

    def test_get_private_key_missing_client(self, key_store):
        """Test getting private key for non-existent client."""
        with pytest.raises(KeyError):
            key_store.get_private_key("nonexistent")

    def test_get_public_key(self, key_store):
        """Test getting public key."""
        key_store.generate_keypair("client_0")

        public_key = key_store.get_public_key("client_0")
        assert isinstance(public_key, ec.EllipticCurvePublicKey)

    def test_get_public_key_string(self, key_store):
        """Test getting public key as string."""
        key_store.generate_keypair("client_0")

        key_str = key_store.get_public_key_string("client_0")
        assert isinstance(key_str, str)
        assert len(key_str) > 0

    def test_remove_client(self, key_store):
        """Test removing client."""
        key_store.generate_keypair("client_0")
        assert key_store.has_client("client_0")

        key_store.remove_client("client_0")
        assert not key_store.has_client("client_0")

    def test_list_clients(self, key_store):
        """Test listing all clients."""
        key_store.generate_keypair("client_0")
        key_store.generate_keypair("client_1")
        key_store.generate_keypair("client_2")

        clients = key_store.list_clients()
        assert len(clients) == 3
        assert "client_0" in clients
        assert "client_1" in clients
        assert "client_2" in clients


class TestKeyManager:
    """Tests for file-based KeyManager."""

    @pytest.fixture
    def key_manager(self, temp_directory):
        """Create key manager instance with temp directory."""
        return KeyManager(keys_dir=temp_directory / "keys")

    def test_generate_and_save_keys(self, key_manager):
        """Test generating and saving keys."""
        private_path, public_path = key_manager.generate_and_save_keys("client_0")

        assert Path(private_path).exists()
        assert Path(public_path).exists()
        assert key_manager.client_has_keys("client_0")

    def test_load_private_key(self, key_manager):
        """Test loading private key."""
        key_manager.generate_and_save_keys("client_0")

        private_key = key_manager.load_private_key("client_0")
        assert isinstance(private_key, ec.EllipticCurvePrivateKey)

    def test_load_private_key_missing(self, key_manager):
        """Test loading private key for non-existent client."""
        with pytest.raises(FileNotFoundError):
            key_manager.load_private_key("nonexistent")

    def test_load_public_key(self, key_manager):
        """Test loading public key."""
        key_manager.generate_and_save_keys("client_0")

        public_key = key_manager.load_public_key("client_0")
        assert isinstance(public_key, ec.EllipticCurvePublicKey)

    def test_load_public_key_string(self, key_manager):
        """Test loading public key as string."""
        key_manager.generate_and_save_keys("client_0")

        key_str = key_manager.load_public_key_string("client_0")
        assert isinstance(key_str, str)
        assert len(key_str) > 0

    def test_delete_keys(self, key_manager):
        """Test deleting keys."""
        key_manager.generate_and_save_keys("client_0")
        assert key_manager.client_has_keys("client_0")

        key_manager.delete_keys("client_0")
        assert not key_manager.client_has_keys("client_0")

    def test_list_clients(self, key_manager):
        """Test listing all clients with keys."""
        key_manager.generate_and_save_keys("client_0")
        key_manager.generate_and_save_keys("client_1")
        key_manager.generate_and_save_keys("client_2")

        clients = key_manager.list_clients()
        assert len(clients) == 3
        assert "client_0" in clients
        assert "client_1" in clients
        assert "client_2" in clients

    def test_rotate_keys(self, key_manager):
        """Test key rotation."""
        key_manager.generate_and_save_keys("client_0")
        
        old_key = key_manager.load_public_key("client_0")
        old_info = key_manager.signature_manager.get_key_info(old_key)

        # Rotate keys
        key_manager.rotate_keys("client_0", backup_old=True)
        
        new_key = key_manager.load_public_key("client_0")
        new_info = key_manager.signature_manager.get_key_info(new_key)

        # New key should be different
        assert old_info["x"] != new_info["x"]
        assert old_info["y"] != new_info["y"]
        
        # Check backup exists (files start with backup prefix)
        backup_files = list(key_manager.keys_dir.glob("*.bak_*"))
        assert len(backup_files) > 0

    def test_generate_keys_with_password(self, key_manager):
        """Test generating encrypted keys."""
        password = b"test_password"
        key_manager.generate_and_save_keys("client_0", password=password)

        # Should load with correct password
        private_key = key_manager.load_private_key("client_0", password=password)
        assert isinstance(private_key, ec.EllipticCurvePrivateKey)

        # Should fail with wrong password
        with pytest.raises(ValueError):
            key_manager.load_private_key("client_0", password=b"wrong_password")


class TestEndToEndCrypto:
    """End-to-end cryptographic tests."""

    def test_full_sign_verify_workflow(self, seed_randomness, sample_model_params):
        """Test complete signing and verification workflow."""
        # Setup
        signature_manager = SignatureManager()
        private_key, public_key = signature_manager.generate_keypair()

        # Create update
        update = ModelUpdate(
            client_id="test_client",
            round_num=1,
            parameters=sample_model_params,
            num_samples=100,
            metrics={"loss": 0.5},
        )

        # Sign
        signature = signature_manager.sign_update(update, private_key)
        public_key_str = signature_manager.serialize_public_key(public_key)

        signed_update = SignedUpdate(
            update=update,
            signature=signature,
            public_key=public_key_str,
        )

        # Verify
        assert signature_manager.verify_update(signed_update) is True
        assert signed_update.is_valid_signature() is True

    def test_multiple_clients_workflow(self, seed_randomness, sample_model_params):
        """Test workflow with multiple clients."""
        key_store = KeyStore()
        signature_manager = SignatureManager()

        # Setup multiple clients
        client_ids = [f"client_{i}" for i in range(5)]
        for client_id in client_ids:
            key_store.generate_keypair(client_id)

        # Sign updates from each client
        signed_updates = []
        for client_id in client_ids:
            update = ModelUpdate(
                client_id=client_id,
                round_num=1,
                parameters=sample_model_params,
                num_samples=100,
            )

            private_key = key_store.get_private_key(client_id)
            signature = signature_manager.sign_update(update, private_key)
            public_key_str = key_store.get_public_key_string(client_id)

            signed_updates.append(SignedUpdate(
                update=update,
                signature=signature,
                public_key=public_key_str,
            ))

        # Verify all signatures
        for signed_update in signed_updates:
            assert signature_manager.verify_update(signed_update) is True
