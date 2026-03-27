"""Unit tests for serving module."""

import pytest
from pathlib import Path
import tempfile
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from src.serving.api import create_app, FraudDetectionAPI
from src.serving.model_store import ModelStore, ModelInfo
from src.serving.prediction import (
    Predictor,
    TransactionData,
    PredictionRequest,
    ModelMetadata,
)
from src.models.lstm import LSTMFraudDetector


@pytest.fixture
def sample_config():
    """Sample model configuration."""
    return {
        "num_features": 30,
        "sequence_length": 10,
        "threshold": 0.5,
        "model_type": "lstm",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "output_size": 2,
    }


@pytest.fixture
def sample_model(sample_config):
    """Create a sample model."""
    model = LSTMFraudDetector(
        input_size=sample_config["num_features"],
        hidden_size=sample_config["hidden_size"],
        num_layers=sample_config["num_layers"],
        dropout=sample_config["dropout"],
        output_size=sample_config["output_size"],
    )
    return model


@pytest.fixture
def temp_model_store(tmp_path):
    """Create a temporary model store."""
    return ModelStore(store_path=tmp_path, max_versions=3)


@pytest.fixture
def sample_transactions():
    """Sample transaction data."""
    return [
        {
            "transaction_id": "txn_001",
            "amount": 150.0,
            "merchant_id": 12345,
            "account_id": 67890,
            "hour": 14,
            "day_of_week": 2,
            "card_present": True,
            "online_transaction": False,
        },
        {
            "transaction_id": "txn_002",
            "amount": 999.99,
            "merchant_id": 54321,
            "account_id": 67890,
            "hour": 3,
            "day_of_week": 5,
            "card_present": False,
            "online_transaction": True,
        },
    ]


class TestPredictor:
    """Tests for Predictor class."""

    def test_from_checkpoint(self, sample_model, sample_config, tmp_path):
        """Test loading predictor from checkpoint."""
        # Save checkpoint
        checkpoint_path = tmp_path / "model.pt"
        torch.save({
            "model_state_dict": sample_model.state_dict(),
            "version": "v1.0",
            "round": 10,
            "metrics": {
                "accuracy": 0.95,
                "auc_roc": 0.98,
            },
            "created_at": "2025-01-01T00:00:00",
        }, checkpoint_path)

        # Load predictor
        predictor = Predictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=sample_config,
            device=torch.device("cpu"),
        )

        assert predictor.metadata.version == "v1.0"
        assert predictor.metadata.training_round == 10
        assert predictor.metadata.auc_roc == 0.98

    def test_predict(self, sample_model, sample_config, tmp_path, sample_transactions):
        """Test making predictions."""
        # Save and load predictor
        checkpoint_path = tmp_path / "model.pt"
        torch.save({
            "model_state_dict": sample_model.state_dict(),
            "version": "v1.0",
        }, checkpoint_path)

        predictor = Predictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=sample_config,
            device=torch.device("cpu"),
        )

        # Make predictions
        predictions = predictor.predict(sample_transactions)

        assert len(predictions) == len(sample_transactions)
        assert "transaction_id" in predictions[0]
        assert "is_fraud" in predictions[0]
        assert "fraud_probability" in predictions[0]
        assert 0 <= predictions[0]["fraud_probability"] <= 1

    def test_predict_single(self, sample_model, sample_config, tmp_path):
        """Test single transaction prediction."""
        checkpoint_path = tmp_path / "model.pt"
        torch.save({
            "model_state_dict": sample_model.state_dict(),
        }, checkpoint_path)

        predictor = Predictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=sample_config,
            device=torch.device("cpu"),
        )

        transaction = {
            "transaction_id": "txn_001",
            "amount": 100.0,
            "merchant_id": 12345,
            "account_id": 67890,
        }

        prediction = predictor.predict_single(transaction)

        assert "transaction_id" in prediction
        assert "is_fraud" in prediction

    def test_update_threshold(self, sample_model, sample_config, tmp_path):
        """Test updating prediction threshold."""
        checkpoint_path = tmp_path / "model.pt"
        torch.save({
            "model_state_dict": sample_model.state_dict(),
        }, checkpoint_path)

        predictor = Predictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=sample_config,
            device=torch.device("cpu"),
        )

        assert predictor.threshold == 0.5

        predictor.update_threshold(0.7)
        assert predictor.threshold == 0.7

        # Invalid threshold should raise error
        with pytest.raises(ValueError):
            predictor.update_threshold(1.5)


class TestModelStore:
    """Tests for ModelStore class."""

    def test_save_model(self, temp_model_store, sample_model, sample_config):
        """Test saving a model."""
        version = temp_model_store.save_model(
            model=sample_model,
            version="v1.0",
            config=sample_config,
        )

        assert version == "v1.0"
        assert version in temp_model_store.models
        assert temp_model_store.models[version].version == "v1.0"

    def test_activate_model(self, temp_model_store, sample_model, sample_config):
        """Test activating a model version."""
        # Save two versions
        v1 = temp_model_store.save_model(sample_model, version="v1.0", config=sample_config)
        v2 = temp_model_store.save_model(sample_model, version="v2.0", config=sample_config)

        # Activate v1
        temp_model_store.activate_model(v1)

        assert temp_model_store.get_active_version() == v1
        assert temp_model_store.models[v1].is_active is True
        assert temp_model_store.models[v2].is_active is False

    def test_get_active_predictor(self, temp_model_store, sample_model, sample_config):
        """Test getting active predictor."""
        # Save and activate model
        version = temp_model_store.save_model(sample_model, version="v1.0", config=sample_config)
        temp_model_store.activate_model(version)

        # Get predictor
        predictor = temp_model_store.get_active_predictor(config=sample_config)

        assert predictor is not None
        assert predictor.metadata.version == version

    def test_rollback(self, temp_model_store, sample_model, sample_config):
        """Test rollback to previous version."""
        # Save multiple versions
        v1 = temp_model_store.save_model(sample_model, version="v1.0", config=sample_config)
        v2 = temp_model_store.save_model(sample_model, version="v2.0", config=sample_config)
        temp_model_store.activate_model(v2)

        # Rollback
        success = temp_model_store.rollback()

        assert success is True
        assert temp_model_store.get_active_version() == v1

    def test_list_versions(self, temp_model_store, sample_model, sample_config):
        """Test listing model versions."""
        # Save multiple versions
        temp_model_store.save_model(sample_model, version="v1.0", config=sample_config)
        temp_model_store.save_model(sample_model, version="v2.0", config=sample_config)

        versions = temp_model_store.list_versions()

        assert len(versions) == 2
        assert any(v["version"] == "v1.0" for v in versions)
        assert any(v["version"] == "v2.0" for v in versions)

    def test_delete_version(self, temp_model_store, sample_model, sample_config):
        """Test deleting a model version."""
        # Save versions
        v1 = temp_model_store.save_model(sample_model, version="v1.0", config=sample_config)
        v2 = temp_model_store.save_model(sample_model, version="v2.0", config=sample_config)
        temp_model_store.activate_model(v2)

        # Try to delete active version (should fail)
        success = temp_model_store.delete_version(v2)
        assert success is False

        # Delete inactive version
        success = temp_model_store.delete_version(v1)
        assert success is True
        assert v1 not in temp_model_store.models

    def test_cleanup_old_versions(self, temp_model_store, sample_model, sample_config):
        """Test automatic cleanup of old versions."""
        # Set max versions to 3
        temp_model_store.max_versions = 3

        # Save 5 versions
        for i in range(5):
            temp_model_store.save_model(
                sample_model,
                version=f"v{i}.0",
                config=sample_config,
            )

        # Should only keep max_versions
        versions = temp_model_store.list_versions()
        assert len(versions) <= 3


class TestFraudDetectionAPI:
    """Tests for FraudDetection API."""

    @pytest.fixture
    def api_client(self, temp_model_store, sample_model, sample_config):
        """Create test client for API."""
        # Save and activate model
        version = temp_model_store.save_model(sample_model, version="v1.0", config=sample_config)
        temp_model_store.activate_model(version)

        # Create app
        app = create_app(
            store=temp_model_store,
            app_config={"model": sample_config, "device": "cpu"},
        )

        return TestClient(app)

    def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_get_model_info(self, api_client):
        """Test getting model info."""
        response = api_client.get("/model/info")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "model_type" in data
        assert "performance" in data

    def test_list_model_versions(self, api_client):
        """Test listing model versions."""
        response = api_client.get("/model/versions")

        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
        assert len(data["versions"]) > 0

    def test_predict(self, api_client, sample_transactions):
        """Test prediction endpoint."""
        request_data = {
            "transactions": sample_transactions,
            "return_probabilities": True,
        }

        response = api_client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "model_info" in data
        assert "processing_time_ms" in data
        assert len(data["predictions"]) == len(sample_transactions)

    def test_predict_single(self, api_client):
        """Test single transaction prediction."""
        transaction = {
            "transaction_id": "txn_001",
            "amount": 100.0,
            "merchant_id": 12345,
            "account_id": 67890,
        }

        response = api_client.post("/predict/single", json=transaction)

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data

    def test_predict_batch(self, api_client, sample_transactions):
        """Test batch prediction."""
        response = api_client.post(
            "/predict/batch",
            json=sample_transactions,
        )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "n_transactions" in data
        assert data["n_transactions"] == len(sample_transactions)

    def test_update_threshold(self, api_client):
        """Test updating prediction threshold."""
        response = api_client.post(
            "/model/threshold",
            params={"threshold": 0.7},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["threshold"] == 0.7

    def test_invalid_threshold(self, api_client):
        """Test invalid threshold value."""
        response = api_client.post(
            "/model/threshold",
            params={"threshold": 1.5},
        )

        assert response.status_code == 422  # Validation error

    def test_empty_transactions(self, api_client):
        """Test prediction with empty transactions list."""
        request_data = {
            "transactions": [],
        }

        response = api_client.post("/predict", json=request_data)

        # Should return validation error
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
