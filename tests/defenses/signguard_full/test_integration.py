"""Integration tests for SignGuard end-to-end FL simulation."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from signguard.core import SignGuardClient, SignGuardServer
from signguard.core.types import ClientConfig, ServerConfig
from signguard.crypto import SignatureManager, KeyStore
from signguard.detection import EnsembleDetector
from signguard.reputation import DecayReputationSystem
from signguard.aggregation import WeightedAggregator


@pytest.fixture
def seed_randomness():
    """Set random seeds."""
    torch.manual_seed(42)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create synthetic dataset
    num_samples = 1000
    input_dim = 28
    num_classes = 2
    
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader


@pytest.fixture
def sample_model():
    """Create sample model - simple 2-layer architecture."""
    return nn.Sequential(
        nn.Linear(28, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


@pytest.fixture
def client_configs():
    """Create client configurations."""
    return [
        ClientConfig(
            client_id=f"client_{i}",
            local_epochs=2,
            learning_rate=0.01,
            batch_size=32,
            device="cpu",
        )
        for i in range(5)
    ]


@pytest.fixture
def key_store():
    """Create in-memory key store."""
    return KeyStore()


@pytest.fixture
def signature_manager():
    """Create signature manager."""
    return SignatureManager()


class TestSignGuardClient:
    """Tests for SignGuardClient."""

    def test_client_creation(
        self,
        sample_model,
        sample_data,
        signature_manager,
        key_store,
    ):
        """Test creating a client."""
        key_store.generate_keypair("client_0")
        private_key = key_store.get_private_key("client_0")
        
        client = SignGuardClient(
            client_id="client_0",
            model=sample_model,
            train_loader=sample_data,
            signature_manager=signature_manager,
            private_key=private_key,
        )
        
        assert client.client_id == "client_0"
        assert client.config.local_epochs == 5

    def test_client_train(
        self,
        sample_model,
        sample_data,
        signature_manager,
        key_store,
    ):
        """Test client training."""
        key_store.generate_keypair("client_0")
        private_key = key_store.get_private_key("client_0")
        
        config = ClientConfig(client_id="client_0", local_epochs=2)
        client = SignGuardClient(
            client_id="client_0",
            model=sample_model,
            train_loader=sample_data,
            signature_manager=signature_manager,
            private_key=private_key,
            config=config,
        )
        
        # Train
        global_params = sample_model.state_dict()
        update = client.train(global_params, current_round=0)
        
        assert update.client_id == "client_0"
        assert update.round_num == 0
        assert update.num_samples > 0
        assert "loss" in update.metrics
        assert "accuracy" in update.metrics

    def test_client_sign_update(
        self,
        sample_model,
        sample_data,
        signature_manager,
        key_store,
    ):
        """Test signing an update."""
        key_store.generate_keypair("client_0")
        private_key = key_store.get_private_key("client_0")
        
        client = SignGuardClient(
            client_id="client_0",
            model=sample_model,
            train_loader=sample_data,
            signature_manager=signature_manager,
            private_key=private_key,
        )
        
        # Create update
        global_params = sample_model.state_dict()
        update = client.train(global_params, current_round=0)
        
        # Sign update
        signed_update = client.sign_update(update)
        
        assert signed_update.signature is not None
        assert len(signed_update.signature) > 0
        assert signed_update.public_key is not None
        assert len(signed_update.public_key) > 0


class TestSignGuardServer:
    """Tests for SignGuardServer."""

    def test_server_creation(
        self,
        sample_model,
        signature_manager,
    ):
        """Test creating a server."""
        server = SignGuardServer(
            global_model=sample_model,
            signature_manager=signature_manager,
        )
        
        assert server.current_round == 0
        assert server.global_model is not None

    def test_verify_signatures(
        self,
        sample_model,
        sample_data,
        signature_manager,
        key_store,
    ):
        """Test signature verification."""
        server = SignGuardServer(
            global_model=sample_model,
            signature_manager=signature_manager,
        )
        
        # Create clients and signed updates
        signed_updates = []
        for i in range(3):
            key_store.generate_keypair(f"client_{i}")
            private_key = key_store.get_private_key(f"client_{i}")
            
            client = SignGuardClient(
                client_id=f"client_{i}",
                model=sample_model,
                train_loader=sample_data,
                signature_manager=signature_manager,
                private_key=private_key,
            )
            
            global_params = sample_model.state_dict()
            update = client.train(global_params, current_round=0)
            signed_update = client.sign_update(update)
            signed_updates.append(signed_update)
        
        # Verify signatures
        verified, rejected = server.verify_signatures(signed_updates)
        
        assert len(verified) == 3
        assert len(rejected) == 0

    def test_detect_anomalies(
        self,
        sample_model,
        signature_manager,
        key_store,
    ):
        """Test anomaly detection."""
        server = SignGuardServer(
            global_model=sample_model,
            signature_manager=signature_manager,
        )
        
        # Create signed updates
        signed_updates = []
        for i in range(3):
            key_store.generate_keypair(f"client_{i}")
            private_key = key_store.get_private_key(f"client_{i}")
            public_key_str = key_store.get_public_key_string(f"client_{i}")
            
            from signguard.core.types import ModelUpdate, SignedUpdate
            
            # Create normal updates
            update = ModelUpdate(
                client_id=f"client_{i}",
                round_num=0,
                parameters=sample_model.state_dict(),
                num_samples=100,
                metrics={"loss": 0.5},
            )
            
            signature = signature_manager.sign_update(update, private_key)
            signed_update = SignedUpdate(
                update=update,
                signature=signature,
                public_key=public_key_str,
            )
            signed_updates.append(signed_update)
        
        # Detect anomalies
        anomaly_scores = server.detect_anomalies(signed_updates)
        
        assert len(anomaly_scores) == 3
        for client_id, score in anomaly_scores.items():
            assert 0.0 <= score.combined_score <= 1.0


class TestEndToEndFederatedLearning:
    """End-to-end federated learning tests."""

    def test_single_round_fl(
        self,
        seed_randomness,
        sample_model,
        sample_data,
        signature_manager,
        key_store,
    ):
        """Test single FL round."""
        # Create server
        server = SignGuardServer(
            global_model=sample_model,
            signature_manager=signature_manager,
        )
        
        # Initialize clients
        num_clients = 5
        server.initialize_clients([f"client_{i}" for i in range(num_clients)])
        
        # Create clients with same model architecture
        signed_updates = []
        for i in range(num_clients):
            key_store.generate_keypair(f"client_{i}")
            private_key = key_store.get_private_key(f"client_{i}")
            
            # Each client gets a fresh model instance
            client_model = nn.Sequential(
                nn.Linear(28, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
            
            client = SignGuardClient(
                client_id=f"client_{i}",
                model=client_model,
                train_loader=sample_data,
                signature_manager=signature_manager,
                private_key=private_key,
                config=ClientConfig(client_id=f"client_{i}", local_epochs=1),
            )
            
            # Train
            global_params = server.get_global_model()
            update = client.train(global_params, current_round=0)
            signed_update = client.sign_update(update)
            signed_updates.append(signed_update)
        
        # Aggregate
        result = server.aggregate(signed_updates)
        
        assert result.round_num == 0
        assert len(result.participating_clients) > 0
        assert result.execution_time > 0
        assert server.current_round == 1

    def test_multi_round_fl(
        self,
        seed_randomness,
        sample_model,
        sample_data,
        signature_manager,
        key_store,
    ):
        """Test multiple FL rounds."""
        # Create server
        server = SignGuardServer(
            global_model=sample_model,
            signature_manager=signature_manager,
            config=ServerConfig(min_clients_required=3),
        )
        
        # Create clients
        num_clients = 3
        client_ids = [f"client_{i}" for i in range(num_clients)]
        server.initialize_clients(client_ids)
        
        clients = []
        for client_id in client_ids:
            key_store.generate_keypair(client_id)
            private_key = key_store.get_private_key(client_id)
            
            # Each client has its own model instance
            client_model = nn.Sequential(
                nn.Linear(28, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
            
            client = SignGuardClient(
                client_id=client_id,
                model=client_model,
                train_loader=sample_data,
                signature_manager=signature_manager,
                private_key=private_key,
                config=ClientConfig(client_id=client_id, local_epochs=1),
            )
            clients.append(client)
        
        # Train for 3 rounds
        num_rounds = 3
        for round_num in range(num_rounds):
            # Collect updates
            signed_updates = []
            for client in clients:
                global_params = server.get_global_model()
                update = client.train(global_params, current_round=round_num)
                signed_update = client.sign_update(update)
                signed_updates.append(signed_update)
            
            # Aggregate
            result = server.aggregate(signed_updates)
            
            assert result.round_num == round_num
        
        assert server.current_round == num_rounds

    def test_with_malicious_client(
        self,
        seed_randomness,
        sample_model,
        sample_data,
        signature_manager,
        key_store,
    ):
        """Test FL with a malicious client."""
        # Create server
        server = SignGuardServer(
            global_model=sample_model,
            signature_manager=signature_manager,
            config=ServerConfig(min_clients_required=3),
        )
        
        # Initialize clients
        num_honest = 4
        num_malicious = 1
        all_clients = [f"client_{i}" for i in range(num_honest + num_malicious)]
        server.initialize_clients(all_clients)
        
        signed_updates = []
        
        # Create honest clients
        for i in range(num_honest):
            key_store.generate_keypair(f"client_{i}")
            private_key = key_store.get_private_key(f"client_{i}")
            
            client_model = nn.Sequential(
                nn.Linear(28, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
            
            client = SignGuardClient(
                client_id=f"client_{i}",
                model=client_model,
                train_loader=sample_data,
                signature_manager=signature_manager,
                private_key=private_key,
                config=ClientConfig(client_id=f"client_{i}", local_epochs=1),
            )
            
            global_params = server.get_global_model()
            update = client.train(global_params, current_round=0)
            signed_update = client.sign_update(update)
            signed_updates.append(signed_update)
        
        # Create malicious client (sends large update)
        malicious_id = f"client_{num_honest}"
        key_store.generate_keypair(malicious_id)
        private_key = key_store.get_private_key(malicious_id)
        public_key_str = key_store.get_public_key_string(malicious_id)
        
        from signguard.core.types import ModelUpdate, SignedUpdate
        
        # Create malicious update (large magnitude)
        malicious_params = {
            name: param + torch.randn_like(param) * 10.0
            for name, param in sample_model.state_dict().items()
        }
        
        malicious_update = ModelUpdate(
            client_id=malicious_id,
            round_num=0,
            parameters=malicious_params,
            num_samples=100,
            metrics={"loss": 5.0},
        )
        
        signature = signature_manager.sign_update(malicious_update, private_key)
        malicious_signed = SignedUpdate(
            update=malicious_update,
            signature=signature,
            public_key=public_key_str,
        )
        signed_updates.append(malicious_signed)
        
        # Aggregate (should detect and exclude malicious)
        result = server.aggregate(signed_updates)
        
        # Check that aggregation succeeded
        assert len(result.participating_clients) > 0

    def test_checkpoint_save_load(
        self,
        sample_model,
        signature_manager,
        tmp_path,
    ):
        """Test saving and loading server checkpoint."""
        server = SignGuardServer(
            global_model=sample_model,
            signature_manager=signature_manager,
        )
        
        # Set some state
        server.current_round = 5
        server.initialize_clients(["client_0", "client_1"])
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pth"
        server.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Create new server with SAME model architecture and load
        new_model = nn.Sequential(
            nn.Linear(28, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        new_server = SignGuardServer(
            global_model=new_model,
            signature_manager=signature_manager,
        )
        
        new_server.load_checkpoint(str(checkpoint_path))
        
        assert new_server.current_round == 5
        assert "client_0" in new_server.get_reputations()
        assert "client_1" in new_server.get_reputations()
