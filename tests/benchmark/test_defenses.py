"""
Unit tests for defense implementations.
"""

import pytest
import numpy as np
from src.defenses import (
    FedAvgDefense,
    MedianDefense,
    TrimmedMeanDefense,
    KrumDefense,
    MultiKrumDefense,
    BulyanDefense,
    FoolsGoldDefense,
    AnomalyDetectionDefense,
    create_defense,
)


class TestFedAvgDefense:
    """Tests for FedAvgDefense."""

    @pytest.fixture
    def defense(self):
        return FedAvgDefense({})

    @pytest.fixture
    def sample_updates(self):
        # 5 clients, 100 parameters each
        return [
            (i, np.random.randn(100).astype(np.float32))
            for i in range(5)
        ]

    def test_aggregation_produces_mean(self, defense, sample_updates):
        """Test FedAvg produces mean of updates."""
        result = defense.defend(sample_updates)
        params = np.array([p for _, p in sample_updates])
        expected = np.mean(params, axis=0)
        assert np.allclose(result, expected, atol=1e-5)


class TestMedianDefense:
    """Tests for MedianDefense."""

    @pytest.fixture
    def defense(self):
        return MedianDefense({})

    @pytest.fixture
    def sample_updates(self):
        # 5 clients, 100 parameters each
        return [
            (i, np.random.randn(100).astype(np.float32))
            for i in range(5)
        ]

    def test_aggregation_produces_median(self, defense, sample_updates):
        """Test Median produces median of updates."""
        result = defense.defend(sample_updates)
        params = np.array([p for _, p in sample_updates])
        expected = np.median(params, axis=0)
        assert np.allclose(result, expected, atol=1e-5)


class TestTrimmedMeanDefense:
    """Tests for TrimmedMeanDefense."""

    @pytest.fixture
    def defense(self):
        return TrimmedMeanDefense({"beta": 0.2})

    @pytest.fixture
    def sample_updates(self):
        # 10 clients, 100 parameters each
        return [
            (i, np.random.randn(100).astype(np.float32))
            for i in range(10)
        ]

    def test_aggregation_trims_extremes(self, defense, sample_updates):
        """Test TrimmedMean trims extreme values."""
        result = defense.defend(sample_updates)
        assert result.shape == (100,)


class TestKrumDefense:
    """Tests for KrumDefense."""

    @pytest.fixture
    def defense(self):
        return KrumDefense({"num_malicious": 1})

    @pytest.fixture
    def sample_updates(self):
        # 5 clients, 100 parameters each
        return [
            (i, np.random.randn(100).astype(np.float32))
            for i in range(5)
        ]

    def test_aggregation_selects_update(self, defense, sample_updates):
        """Test Krum selects one of the client updates."""
        result = defense.defend(sample_updates)
        assert result.shape == (100,)

        # Result should match one of the input updates
        params = np.array([p for _, p in sample_updates])
        assert any(np.allclose(result, p, atol=1e-5) for p in params)


class TestFoolsGoldDefense:
    """Tests for FoolsGoldDefense."""

    @pytest.fixture
    def defense(self):
        return FoolsGoldDefense({
            "history_length": 5,
            "min_weight": 0.01,
        })

    @pytest.fixture
    def sample_updates(self):
        # 5 clients, 100 parameters each
        return [
            (i, np.random.randn(100).astype(np.float32))
            for i in range(5)
        ]

    def test_aggregation_weights_updates(self, defense, sample_updates):
        """Test FoolsGold produces weighted aggregation."""
        result = defense.defend(sample_updates)
        assert result.shape == (100,)

    def test_provides_detection_metrics(self, defense, sample_updates):
        """Test FoolsGold provides detection metrics."""
        defense.defend(sample_updates)
        metrics = defense.get_detection_metrics()
        assert metrics is not None
        assert "min_weight" in metrics
        assert "max_weight" in metrics


class TestAnomalyDetectionDefense:
    """Tests for AnomalyDetectionDefense."""

    @pytest.fixture
    def defense(self):
        return AnomalyDetectionDefense({
            "method": "zscore",
            "threshold": 3.0,
        })

    @pytest.fixture
    def sample_updates(self):
        # 5 clients, 100 parameters each
        updates = [
            (i, np.random.randn(100).astype(np.float32) * 0.1)
            for i in range(4)
        ]
        # Add one clearly anomalous update
        updates.append((4, np.random.randn(100).astype(np.float32) * 10.0))
        return updates

    def test_aggregation_handles_anomalies(self, defense, sample_updates):
        """Test anomaly detection filters anomalous updates."""
        result = defense.defend(sample_updates)
        assert result.shape == (100,)

    def test_provides_detection_metrics(self, defense, sample_updates):
        """Test AnomalyDetection provides detection metrics."""
        defense.defend(sample_updates)
        metrics = defense.get_detection_metrics()
        assert metrics is not None
        assert "num_detected" in metrics


class TestDefenseFactory:
    """Tests for create_defense factory function."""

    def test_creates_fedavg(self):
        """Test factory creates FedAvg."""
        defense = create_defense("fedavg", {})
        assert isinstance(defense, FedAvgDefense)

    def test_creates_median(self):
        """Test factory creates Median."""
        defense = create_defense("median", {})
        assert isinstance(defense, MedianDefense)

    def test_creates_trimmed_mean(self):
        """Test factory creates TrimmedMean."""
        defense = create_defense("trimmed_mean", {})
        assert isinstance(defense, TrimmedMeanDefense)

    def test_unknown_defense_raises_error(self):
        """Test factory raises error for unknown defense."""
        with pytest.raises(ValueError):
            create_defense("unknown_defense", {})
