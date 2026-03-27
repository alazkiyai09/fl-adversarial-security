"""Unit tests for federated learning module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from omegaconf import OmegaConf

from src.fl.client import FlowerClient, create_client
from src.fl.server import FlowerServer, SimulationServer
from src.fl.strategy import create_strategy, weighted_average
from src.fl.defenses.signguard import SignGuardDefense, KrumDefense


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    config_dict = {
        "project": {"random_seed": 42},
        "data": {
            "batch_size": 32,
            "train_split": 0.8,
            "val_split": 0.1,
            "n_clients": 3,
            "num_features": 10,
            "sequence_length": 5,
        },
        "model": {
            "type": "lstm",
            "input_size": 10,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.1,
            "output_size": 2,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "weight_decay": 1e-5,
            "scheduler": "none",
            "loss_function": "cross_entropy",
        },
        "fl": {
            "server_address": "localhost",
            "server_port": 8080,
            "n_rounds": 2,
            "local_epochs": 1,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "min_available_clients": 2,
            "checkpoint_dir": "/tmp/test_checkpoints",
        },
        "strategy": {
            "name": "fedavg",
        },
        "privacy": {
            "dp_enabled": False,
            "epsilon": 1.0,
            "delta": 1e-5,
            "noise_multiplier": 0.1,
            "max_grad_norm": 1.0,
            "secure_agg_enabled": False,
        },
        "security": {
            "signguard_enabled": False,
            "signguard_threshold": 0.1,
        },
        "mlops": {
            "mlflow_enabled": False,
            "mlflow_experiment_name": "test",
        },
        "device": "cpu",
        "num_workers": 0,
        "pin_memory": False,
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def sample_model(sample_config):
    """Create sample model."""
    from src.models.lstm import LSTMFraudDetector

    model = LSTMFraudDetector(
        input_size=sample_config.model.input_size,
        hidden_size=sample_config.model.hidden_size,
        num_layers=sample_config.model.num_layers,
        dropout=sample_config.model.dropout,
        output_size=sample_config.model.output_size,
    )
    return model


@pytest.fixture
def sample_data_loaders():
    """Create sample data loaders."""
    # Create dummy data
    X_train = torch.randn(100, 5, 10)  # 100 samples, seq_length=5, features=10
    y_train = torch.randint(0, 2, (100,))

    X_test = torch.randn(20, 5, 10)
    y_test = torch.randint(0, 2, (20,))

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, test_loader


class TestFlowerClient:
    """Tests for FlowerClient."""

    def test_client_initialization(self, sample_config, sample_model, sample_data_loaders):
        """Test client initialization."""
        train_loader, test_loader = sample_data_loaders

        client = FlowerClient(
            model=sample_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=sample_config,
            client_id=0,
            device=torch.device("cpu"),
        )

        assert client.client_id == 0
        assert client.local_epochs == sample_config.fl.local_epochs
        assert client.learning_rate == sample_config.model.learning_rate
        assert len(train_loader) > 0
        assert len(test_loader) > 0

    def test_get_parameters(self, sample_config, sample_model, sample_data_loaders):
        """Test getting model parameters."""
        train_loader, test_loader = sample_data_loaders

        client = FlowerClient(
            model=sample_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=sample_config,
            client_id=0,
            device=torch.device("cpu"),
        )

        params = client.get_parameters(config={})

        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_parameters(self, sample_config, sample_model, sample_data_loaders):
        """Test setting model parameters."""
        train_loader, test_loader = sample_data_loaders

        client = FlowerClient(
            model=sample_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=sample_config,
            client_id=0,
            device=torch.device("cpu"),
        )

        # Get parameters
        params = client.get_parameters(config={})

        # Set parameters back
        client.set_parameters(params)

        # Verify model still works
        client.model.eval()
        with torch.no_grad():
            for data, _ in train_loader:
                output = client.model(data)
                assert output is not None
                break

    def test_fit(self, sample_config, sample_model, sample_data_loaders):
        """Test client training."""
        train_loader, test_loader = sample_data_loaders

        client = FlowerClient(
            model=sample_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=sample_config,
            client_id=0,
            device=torch.device("cpu"),
        )

        # Get initial parameters
        initial_params = client.get_parameters(config={})

        # Train
        updated_params, num_examples, metrics = client.fit(
            parameters=initial_params,
            config={"round": 1},
        )

        assert isinstance(updated_params, list)
        assert num_examples > 0
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_evaluate(self, sample_config, sample_model, sample_data_loaders):
        """Test client evaluation."""
        train_loader, test_loader = sample_data_loaders

        client = FlowerClient(
            model=sample_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=sample_config,
            client_id=0,
            device=torch.device("cpu"),
        )

        params = client.get_parameters(config={})

        loss, num_examples, metrics = client.evaluate(
            parameters=params,
            config={},
        )

        assert isinstance(loss, float)
        assert num_examples > 0
        assert "accuracy" in metrics


class TestStrategy:
    """Tests for FL strategy factory."""

    def test_create_fedavg_strategy(self, sample_config):
        """Test FedAvg strategy creation."""
        strategy = create_strategy("fedavg", sample_config)

        assert strategy is not None
        # Check that it's FedAvg or FedAvgWithDefense
        strategy_type = type(strategy).__name__
        assert "FedAvg" in strategy_type

    def test_weighted_average(self):
        """Test weighted average aggregation."""
        metrics = [
            (10, {"accuracy": 0.8, "loss": 0.5}),
            (20, {"accuracy": 0.9, "loss": 0.3}),
            (30, {"accuracy": 0.85, "loss": 0.4}),
        ]

        result = weighted_average(metrics)

        # Compute expected weighted average
        total_examples = 10 + 20 + 30
        expected_acc = (10 * 0.8 + 20 * 0.9 + 30 * 0.85) / total_examples
        expected_loss = (10 * 0.5 + 20 * 0.3 + 30 * 0.4) / total_examples

        assert abs(result["accuracy"] - expected_acc) < 1e-6
        assert abs(result["loss"] - expected_loss) < 1e-6


class TestSignGuardDefense:
    """Tests for SignGuard defense."""

    def test_defense_initialization(self):
        """Test defense initialization."""
        defense = SignGuardDefense(threshold=0.1)

        assert defense.threshold == 0.1
        assert len(defense.sign_similarity_history) == 0

    def test_filter_updates_all_benign(self):
        """Test filtering with all benign updates."""
        defense = SignGuardDefense(threshold=0.1)

        # Create similar updates (benign)
        updates = []
        for _ in range(5):
            update = [torch.randn(10, 10) * 0.1]  # Small random updates
            updates.append(update)

        filtered, scores = defense.filter_updates(updates)

        # Should keep most or all updates
        assert len(filtered) > 0
        assert len(filtered) <= len(updates)
        assert scores is not None
        assert len(scores) == len(updates)

    def test_filter_updates_with_malicious(self):
        """Test filtering with some malicious updates."""
        defense = SignGuardDefense(threshold=0.5)

        updates = []

        # 4 benign updates (similar)
        for _ in range(4):
            update = [torch.randn(10, 10) * 0.1]
            updates.append(update)

        # 1 malicious update (very different)
        updates.append([torch.randn(10, 10) * 10])

        filtered, scores = defense.filter_updates(updates)

        # Should filter out at least the malicious one
        assert len(filtered) < len(updates)
        # Malicious should have lower score
        assert scores[-1] < scores[0]


class TestFlowerServer:
    """Tests for FlowerServer."""

    def test_server_initialization(self, sample_config):
        """Test server initialization."""
        server = FlowerServer(config=sample_config)

        assert server.config == sample_config
        assert server.n_rounds == sample_config.fl.n_rounds
        assert server.strategy is not None

    def test_get_config_params(self, sample_config):
        """Test configuration parameter extraction."""
        server = FlowerServer(config=sample_config)

        params = server._get_config_params()

        assert "n_rounds" in params
        assert "n_clients" in params
        assert "model_type" in params
        assert "strategy" in params
        assert params["n_rounds"] == sample_config.fl.n_rounds


class TestSimulationServer:
    """Tests for SimulationServer."""

    def test_simulation_initialization(self, sample_config, sample_model, sample_data_loaders):
        """Test simulation server initialization."""
        # Create clients
        train_loader, test_loader = sample_data_loaders
        clients = []

        for i in range(3):
            # Clone model for each client
            import copy
            model_copy = type(sample_model)(
                input_size=sample_config.model.input_size,
                hidden_size=sample_config.model.hidden_size,
                num_layers=sample_config.model.num_layers,
                dropout=sample_config.model.dropout,
                output_size=sample_config.model.output_size,
            )

            client = FlowerClient(
                model=model_copy,
                train_loader=train_loader,
                test_loader=test_loader,
                config=sample_config,
                client_id=i,
                device=torch.device("cpu"),
            )
            clients.append(client)

        server = SimulationServer(config=sample_config, clients=clients)

        assert len(server.clients) == 3
        assert server.n_rounds == sample_config.fl.n_rounds
        assert server.strategy is not None

    @pytest.mark.slow
    def test_simulation_run(self, sample_config, sample_model, sample_data_loaders):
        """Test running a short simulation."""
        # Create clients
        train_loader, test_loader = sample_data_loaders
        clients = []

        from src.models.lstm import LSTMFraudDetector

        for i in range(3):
            model = LSTMFraudDetector(
                input_size=sample_config.model.input_size,
                hidden_size=sample_config.model.hidden_size,
                num_layers=sample_config.model.num_layers,
                dropout=sample_config.model.dropout,
                output_size=sample_config.model.output_size,
            )

            client = FlowerClient(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                config=sample_config,
                client_id=i,
                device=torch.device("cpu"),
            )
            clients.append(client)

        # Run short simulation
        server = SimulationServer(config=sample_config, clients=clients)

        # Temporarily disable MLflow
        server.mlflow_tracker = None

        history = server.run()

        # Check that we have metrics for each round
        assert "loss" in history
        assert "accuracy" in history
        assert len(history["loss"]) == sample_config.fl.n_rounds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
