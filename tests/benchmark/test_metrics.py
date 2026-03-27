"""
Unit tests for metrics computation.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.metrics import (
    compute_clean_accuracy,
    compute_auprc,
    compute_attack_success_rate,
    compute_fraud_detection_metrics,
    MetricsHistory,
)


class TestComputeCleanAccuracy:
    """Tests for compute_clean_accuracy."""

    @pytest.fixture
    def model(self):
        """Simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )

    @pytest.fixture
    def test_loader(self):
        """Create test data loader."""
        # Create balanced test data
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=10)

    def test_accuracy_in_range(self, model, test_loader):
        """Test accuracy is between 0 and 1."""
        acc = compute_clean_accuracy(model, test_loader)
        assert 0.0 <= acc <= 1.0

    def test_accuracy_deterministic_with_seed(self, model, test_loader):
        """Test accuracy is deterministic with same seed."""
        torch.manual_seed(42)
        acc1 = compute_clean_accuracy(model, test_loader)

        torch.manual_seed(42)
        acc2 = compute_clean_accuracy(model, test_loader)

        assert acc1 == acc2


class TestComputeAUPRC:
    """Tests for compute_auprc."""

    @pytest.fixture
    def model(self):
        """Simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )

    @pytest.fixture
    def test_loader(self):
        """Create test data loader."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=10)

    def test_auprc_in_range(self, model, test_loader):
        """Test AUPRC is between 0 and 1."""
        auprc = compute_auprc(model, test_loader)
        assert 0.0 <= auprc <= 1.0


class TestComputeAttackSuccessRate:
    """Tests for compute_attack_success_rate."""

    @pytest.fixture
    def model(self):
        """Simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )

    @pytest.fixture
    def test_loader(self):
        """Create test data loader."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=10)

    def test_asr_in_range(self, model, test_loader):
        """Test ASR is between 0 and 1."""
        asr = compute_attack_success_rate(model, test_loader, target_class=0)
        assert 0.0 <= asr <= 1.0


class TestMetricsHistory:
    """Tests for MetricsHistory class."""

    @pytest.fixture
    def history(self):
        return MetricsHistory()

    def test_initialization(self, history):
        """Test history initializes correctly."""
        assert len(history.history["round"]) == 0

    def test_add_metrics(self, history):
        """Test adding metrics updates history."""
        history.add_metrics(
            round_num=1,
            train_loss=0.5,
            train_accuracy=0.8,
            test_accuracy=0.75,
            auprc=0.7,
            asr=0.1,
        )
        assert len(history.history["round"]) == 1
        assert history.history["round"][0] == 1
        assert history.history["train_accuracy"][0] == 0.8

    def test_get_final_metrics(self, history):
        """Test getting final metrics."""
        history.add_metrics(
            round_num=1,
            train_loss=0.5,
            train_accuracy=0.8,
            test_accuracy=0.75,
            auprc=0.7,
            asr=0.1,
        )
        history.add_metrics(
            round_num=2,
            train_loss=0.4,
            train_accuracy=0.85,
            test_accuracy=0.8,
            auprc=0.75,
            asr=0.05,
        )

        final = history.get_final_metrics()
        assert final["round"] == 2
        assert final["test_accuracy"] == 0.8

    def test_get_metric_series(self, history):
        """Test getting metric series."""
        history.add_metrics(
            round_num=1,
            train_loss=0.5,
            train_accuracy=0.8,
            test_accuracy=0.75,
            auprc=0.7,
            asr=0.1,
        )

        series = history.get_metric_series("train_accuracy")
        assert len(series) == 1
        assert series[0] == 0.8

    def test_compute_convergence_round(self, history):
        """Test convergence detection."""
        # Add metrics that converge
        for i in range(5):
            history.add_metrics(
                round_num=i,
                train_loss=0.5 - i * 0.1,
                train_accuracy=0.7 + i * 0.02,
                test_accuracy=0.75 + i * 0.01,
                auprc=0.7 + i * 0.01,
                asr=0.1,
            )

        # Should detect convergence
        conv_round = history.compute_convergence_round(
            metric="test_accuracy",
            threshold=0.05,
            window=3,
        )
        # First 3 rounds should be within threshold
        assert conv_round is not None
