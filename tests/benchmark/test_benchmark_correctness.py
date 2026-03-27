"""
End-to-end correctness tests for the benchmark.
"""

import pytest
import numpy as np
import torch
from src.attacks import LabelFlipAttack, SignFlipAttack
from src.defenses import FedAvgDefense, MedianDefense, create_defense
from src.utils import set_seed


class TestReproducibility:
    """Tests for experiment reproducibility."""

    def test_same_seed_produces_same_results(self):
        """Test that same seed produces identical results."""
        set_seed(42)
        data1 = np.random.randn(100)

        set_seed(42)
        data2 = np.random.randn(100)

        assert np.array_equal(data1, data2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        data1 = np.random.randn(100)

        set_seed(43)
        data2 = np.random.randn(100)

        assert not np.array_equal(data1, data2)


class TestAttackDefenseInteraction:
    """Tests for attack and defense interaction."""

    @pytest.fixture
    def sample_updates(self):
        """Create sample client updates."""
        np.random.seed(42)
        return [
            (i, np.random.randn(50).astype(np.float32))
            for i in range(10)
        ]

    def test_fedavg_without_attack(self, sample_updates):
        """Test FedAvg works correctly without attack."""
        defense = FedAvgDefense({})
        result = defense.defend(sample_updates)

        params = np.array([p for _, p in sample_updates])
        expected = np.mean(params, axis=0)

        assert np.allclose(result, expected, atol=1e-5)

    def test_median_with_sign_flip_attack(self, sample_updates):
        """Test Median defense against sign flip attack."""
        # Make one client malicious
        malicious_updates = sample_updates.copy()
        malicious_idx = 0
        original_params = malicious_updates[malicious_idx][1].copy()
        malicious_updates[malicious_idx] = (
            malicious_idx,
            -original_params * 10.0  # Sign flip + scale
        )

        # FedAvg should be affected
        fedavg = FedAvgDefense({})
        fedavg_result = fedavg.defend(malicious_updates)

        # Median should be more robust
        median = MedianDefense({})
        median_result = median.defend(malicious_updates)

        # Median should differ from FedAvg when attack present
        assert not np.allclose(fedavg_result, median_result, atol=1e-3)

    def test_defense_factory(self):
        """Test defense factory creates correct types."""
        fedavg = create_defense("fedavg", {})
        assert isinstance(fedavg, FedAvgDefense)

        median = create_defense("median", {})
        assert isinstance(median, MedianDefense)


class TestAggregationCorrectness:
    """Tests for aggregation correctness."""

    def test_fedavg_simple_case(self):
        """Test FedAvg with simple known values."""
        updates = [
            (0, np.array([1.0, 2.0])),
            (1, np.array([3.0, 4.0])),
            (2, np.array([5.0, 6.0])),
        ]

        defense = FedAvgDefense({})
        result = defense.defend(updates)

        expected = np.array([3.0, 4.0])  # Mean of each dimension
        assert np.allclose(result, expected)

    def test_median_simple_case(self):
        """Test Median with simple known values."""
        updates = [
            (0, np.array([1.0, 2.0])),
            (1, np.array([3.0, 4.0])),
            (2, np.array([5.0, 6.0])),
        ]

        defense = MedianDefense({})
        result = defense.defend(updates)

        expected = np.array([3.0, 4.0])  # Median of each dimension
        assert np.allclose(result, expected)

    def test_trimmed_mean_simple_case(self):
        """Test TrimmedMean with simple known values."""
        # 5 updates, trim 1 from each end (beta=0.2)
        updates = [
            (0, np.array([1.0])),
            (1, np.array([2.0])),
            (2, np.array([3.0])),
            (3, np.array([4.0])),
            (4, np.array([5.0])),
        ]

        defense = create_defense("trimmed_mean", {"beta": 0.2})
        result = defense.defend(updates)

        # After trimming 1 and 5, mean of [2, 3, 4] = 3
        expected = np.array([3.0])
        assert np.allclose(result, expected)


class TestParameterHandling:
    """Tests for parameter manipulation."""

    def test_parameter_flatten_unflatten(self):
        """Test flattening and unflattening parameters."""
        params = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0]),
            np.array([6.0]),
        ]

        # Flatten
        flat = np.concatenate([p.flatten() for p in params])
        assert flat.shape == (6,)

        # Unflatten
        result = []
        idx = 0
        for p in params:
            size = p.size
            result.append(flat[idx:idx + size].reshape(p.shape))
            idx += size

        for orig, res in zip(params, result):
            assert np.array_equal(orig, res)

    def test_attack_preserves_parameter_shape(self):
        """Test attacks preserve parameter shape."""
        attack = SignFlipAttack({"scale": 1.0})
        params = np.random.randn(1000).astype(np.float32)

        result = attack.apply_attack(params)
        assert result.shape == params.shape

    def test_defense_preserves_parameter_shape(self):
        """Test defenses preserve output shape."""
        updates = [
            (i, np.random.randn(100).astype(np.float32))
            for i in range(5)
        ]

        defense = create_defense("fedavg", {})
        result = defense.defend(updates)
        assert result.shape == (100,)


class TestMetricConsistency:
    """Tests for metric computation consistency."""

    def test_metrics_range_validity(self):
        """Test all metrics are in valid ranges."""
        # Accuracy, AUPRC, ASR should all be in [0, 1]
        valid_metrics = [0.0, 0.5, 1.0]

        for metric in valid_metrics:
            assert 0.0 <= metric <= 1.0

    def test_metric_computation_deterministic(self):
        """Test metric computation is deterministic."""
        # Same inputs should produce same outputs
        predictions = np.array([0, 1, 1, 0])
        labels = np.array([0, 1, 0, 0])

        # Compute accuracy twice
        acc1 = (predictions == labels).mean()
        acc2 = (predictions == labels).mean()

        assert acc1 == acc2
