"""
Integration Tests for SignGuard Flower Integration

Tests client, server, and strategy integration.
"""

import pytest
import time
import numpy as np
import torch
import torch.nn as nn

from src.integration.client import SignGuardClient
from src.integration.server import SignGuardServer
from src.integration.strategy import SignGuardStrategy, create_signguard_strategy
from src.utils.model_utils import create_simple_mlp, get_model_parameters
from src.utils.data_utils import create_dummy_data


class TestSignGuardServer:
    """Test SignGuardServer functionality."""

    def test_register_client(self):
        """Test client registration."""
        server = SignGuardServer()

        # Generate key pair
        from src.crypto.key_manager import generate_key_pair
        _, public_key_pem = generate_key_pair()

        # Register client
        server.register_client("client_1", public_key_pem)

        assert "client_1" in server.get_reputations()

    def test_verify_updates_valid(self):
        """Test signature verification with valid signatures."""
        server = SignGuardServer()

        # Generate keys and sign update
        from src.crypto.key_manager import generate_key_pair
        from src.crypto.signature_handler import SignatureHandler

        private_pem, public_pem = generate_key_pair()
        server.register_client("client_1", public_pem)

        handler = SignatureHandler()
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend

        private_key = serialization.load_pem_private_key(
            private_pem, password=None, backend=default_backend()
        )

        update = [np.random.randn(100).astype(np.float32)]
        signature = handler.sign_parameters(private_key, update, 0, time.time())

        # Verify
        signed_updates = [
            ("client_1", update, signature.hex(), time.time())
        ]

        valid, invalid = server.verify_updates(signed_updates)

        assert len(valid) == 1
        assert len(invalid) == 0
        assert "client_1" in valid

    def test_verify_updates_invalid(self):
        """Test signature verification with invalid signatures."""
        server = SignGuardServer()

        # Register client
        from src.crypto.key_manager import generate_key_pair
        _, public_pem = generate_key_pair()
        server.register_client("client_1", public_pem)

        # Create fake signature
        update = [np.random.randn(100).astype(np.float32)]
        fake_signature = b'\x00' * 64

        # Verify
        signed_updates = [
            ("client_1", update, fake_signature.hex(), time.time())
        ]

        valid, invalid = server.verify_updates(signed_updates)

        assert len(valid) == 0
        assert len(invalid) == 1
        assert "client_1" in invalid

    def test_detect_anomalies(self):
        """Test anomaly detection."""
        server = SignGuardServer()

        # Create updates (10 normal, 1 anomalous)
        updates = {}
        for i in range(10):
            if i == 5:
                # Anomalous
                updates[f"client_{i}"] = [np.random.randn(100) * 5.0]
            else:
                # Normal
                updates[f"client_{i}"] = [np.random.randn(100) * 0.1]

        # Detect
        results = server.detect_anomalies(updates)

        assert len(results) == 10
        # Client 5 should be flagged
        assert results["client_5"]["is_anomalous"] == True

    def test_aggregate_updates(self):
        """Test update aggregation."""
        server = SignGuardServer()

        # Create updates with different reputations
        server.reputation_manager.register_client("client_1")
        server.reputation_manager.update_reputation("client_1", 0.1)  # High rep

        server.reputation_manager.register_client("client_2")
        server.reputation_manager.update_reputation("client_2", 0.9)  # Low rep

        updates = {
            "client_1": [np.array([1.0, 2.0, 3.0])],
            "client_2": [np.array([4.0, 5.0, 6.0])]
        }

        aggregated, metadata = server.aggregate_updates(updates)

        # Should be weighted toward client_1
        assert aggregated[0][0] < 2.5  # Weighted average

    def test_process_round(self):
        """Test complete round processing."""
        server = SignGuardServer()

        # Setup clients
        from src.crypto.key_manager import generate_key_pair
        from src.crypto.signature_handler import SignatureHandler
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend

        handler = SignatureHandler()

        signed_updates = []
        for i in range(5):
            # Generate keys
            private_pem, public_pem = generate_key_pair()
            server.register_client(f"client_{i}", public_pem)

            private_key = serialization.load_pem_private_key(
                private_pem, password=None, backend=default_backend()
            )

            # Create update
            update = [np.random.randn(50).astype(np.float32)]
            signature = handler.sign_parameters(private_key, update, 0, time.time())

            signed_updates.append((
                f"client_{i}",
                update,
                signature.hex(),
                time.time(),
                100
            ))

        # Process round
        aggregated, metadata = server.process_round(signed_updates)

        assert 'num_clients' in metadata
        assert metadata['num_clients'] == 5
        assert len(aggregated) == 1


class TestSignGuardStrategy:
    """Test SignGuardStrategy functionality."""

    def test_create_strategy(self):
        """Test strategy creation."""
        config = {
            'federated_learning': {
                'clients_per_round': 5,
                'fraction_fit': 0.5,
                'fraction_evaluate': 0.2
            }
        }

        initial_params = [np.random.randn(100, 50).astype(np.float32)]
        strategy = create_signguard_strategy(config, initial_params)

        assert strategy.min_fit_clients == 5
        assert strategy.fraction_fit == 0.5

    def test_get_server(self):
        """Test getting server instance."""
        strategy = SignGuardStrategy()
        server = strategy.get_server()

        assert server is not None
        assert isinstance(server, SignGuardServer)

    def test_get_reputations(self):
        """Test getting reputations."""
        strategy = SignGuardStrategy()
        server = strategy.get_server()

        # Register clients
        from src.crypto.key_manager import generate_key_pair
        _, public_key_pem = generate_key_pair()
        server.register_client("client_1", public_key_pem)

        reputations = strategy.get_reputations()

        assert "client_1" in reputations


class TestSignGuardClient:
    """Test SignGuardClient functionality."""

    def test_client_creation(self):
        """Test client creation."""
        # Create model and data
        model = create_simple_mlp()
        train_loader, test_loader = create_dummy_data(num_samples=100)

        # Create client
        client = SignGuardClient(
            client_id="test_client",
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device="cpu",
            key_dir=None
        )

        assert client.client_id == "test_client"
        assert client.get_public_key() is not None

    def test_get_parameters(self):
        """Test getting model parameters."""
        model = create_simple_mlp()
        train_loader, test_loader = create_dummy_data(num_samples=100)

        client = SignGuardClient(
            client_id="test_client",
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device="cpu"
        )

        params = client.get_parameters({})

        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_parameters(self):
        """Test setting model parameters."""
        model = create_simple_mlp()
        train_loader, test_loader = create_dummy_data(num_samples=100)

        client = SignGuardClient(
            client_id="test_client",
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device="cpu"
        )

        # Get parameters, modify, set back
        params = client.get_parameters({})
        params[0] += 0.1

        client.set_parameters(params)

        # Verify change
        new_params = client.get_parameters({})
        np.testing.assert_array_almost_equal(new_params[0], params[0])

    def test_sign_update(self):
        """Test update signing."""
        model = create_simple_mlp()
        train_loader, test_loader = create_dummy_data(num_samples=100)

        client = SignGuardClient(
            client_id="test_client",
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device="cpu"
        )

        update = client.get_parameters({})
        signature = client._sign_update(update, round_num=0)

        assert len(signature) > 0
        assert isinstance(signature, bytes)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_simple_fl_round(self):
        """Test a simple FL round with SignGuard."""
        # Create server
        server = SignGuardServer()

        # Setup crypto
        from src.crypto.key_manager import generate_key_pair
        from src.crypto.signature_handler import SignatureHandler
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend

        handler = SignatureHandler()

        # Create client updates
        signed_updates = []
        for i in range(5):
            # Generate keys
            private_pem, public_pem = generate_key_pair()
            server.register_client(f"client_{i}", public_pem)

            private_key = serialization.load_pem_private_key(
                private_pem, password=None, backend=default_backend()
            )

            # Create update
            update = [np.random.randn(100).astype(np.float32) * 0.1]
            signature = handler.sign_parameters(private_key, update, 0, time.time())

            signed_updates.append((
                f"client_{i}",
                update,
                signature.hex(),
                time.time(),
                100
            ))

        # Process round
        aggregated, metadata = server.process_round(signed_updates)

        # Check results
        assert metadata['num_clients'] == 5
        assert metadata['valid_clients'] == 5
        assert len(aggregated) == 1

    def test_reputation_evolution(self):
        """Test reputation evolution over multiple rounds."""
        server = SignGuardServer()

        # Register clients
        from src.crypto.key_manager import generate_key_pair
        _, public_pem = generate_key_pair()
        server.register_client("good_client", public_pem)
        server.register_client("bad_client", public_pem)

        # Simulate multiple rounds
        for round_num in range(5):
            # Good client: normal updates
            server.reputation_manager.update_reputation("good_client", 0.1)

            # Bad client: anomalous updates
            server.reputation_manager.update_reputation("bad_client", 0.9)

        # Check reputations
        reputations = server.get_reputations()

        assert reputations["good_client"] > reputations["bad_client"]
        assert reputations["good_client"] > 0.6
        assert reputations["bad_client"] < 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
