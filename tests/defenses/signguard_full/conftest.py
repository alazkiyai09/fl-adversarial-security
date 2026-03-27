"""Pytest configuration and fixtures for SignGuard tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict

from signguard.core.types import ModelUpdate, SignedUpdate, ClientConfig, ServerConfig


@pytest.fixture
def seed_randomness():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def sample_model_params() -> Dict[str, torch.Tensor]:
    """Create sample model parameters for testing."""
    return {
        "layer1.weight": torch.randn(128, 28),
        "layer1.bias": torch.randn(128),
        "layer2.weight": torch.randn(64, 128),
        "layer2.bias": torch.randn(64),
        "output.weight": torch.randn(2, 64),
        "output.bias": torch.randn(2),
    }


@pytest.fixture
def sample_model_update(sample_model_params) -> ModelUpdate:
    """Create a sample model update."""
    return ModelUpdate(
        client_id="client_0",
        round_num=1,
        parameters=sample_model_params,
        num_samples=100,
        metrics={"loss": 0.5, "accuracy": 0.9},
    )


@pytest.fixture
def sample_signed_update(sample_model_update) -> SignedUpdate:
    """Create a sample signed update (with mock signature)."""
    return SignedUpdate(
        update=sample_model_update,
        signature="mock_signature_data",
        public_key="mock_public_key_data",
        algorithm="ECDSA",
    )


@pytest.fixture
def client_config() -> ClientConfig:
    """Create a client configuration."""
    return ClientConfig(
        client_id="test_client",
        local_epochs=5,
        learning_rate=0.01,
        batch_size=32,
        device="cpu",
    )


@pytest.fixture
def server_config() -> ServerConfig:
    """Create a server configuration."""
    return ServerConfig(
        num_rounds=100,
        num_clients_per_round=10,
        min_clients_required=5,
        anomaly_threshold=0.7,
        min_reputation_threshold=0.1,
    )


@pytest.fixture
def temp_directory(tmp_path: Path) -> Path:
    """Create a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def mock_clients_list() -> list[str]:
    """Create a list of mock client IDs."""
    return [f"client_{i}" for i in range(20)]


@pytest.fixture
def honest_byzantine_split(mock_clients_list) -> tuple[list[str], list[str]]:
    """Split clients into honest and Byzantine."""
    honest = mock_clients_list[:16]
    byzantine = mock_clients_list[16:]
    return honest, byzantine
