"""
Unit Tests for SignGuard Reputation System

Tests reputation management and weighted aggregation.
"""

import pytest
import numpy as np

from src.reputation.reputation_manager import ReputationManager, update_reputation
from src.reputation.weighted_aggregator import (
    WeightedAggregator,
    aggregate_updates,
    compute_weights
)


class TestReputationManager:
    """Test ReputationManager functionality."""

    def test_register_client(self):
        """Test client registration."""
        rm = ReputationManager()
        rm.register_client("client_1")

        assert rm.get_reputation("client_1") == 0.5  # Initial
        assert "client_1" in rm.get_all_reputations()

    def test_update_reputation_normal(self):
        """Test reputation update with normal behavior."""
        rm = ReputationManager()
        rm.register_client("client_1")

        # Low anomaly score (normal)
        new_rep = rm.update_reputation("client_1", anomaly_score=0.1)

        # Reputation should increase
        assert new_rep > 0.5
        assert new_rep <= 1.0

    def test_update_reputation_anomalous(self):
        """Test reputation update with anomalous behavior."""
        rm = ReputationManager()
        rm.register_client("client_1")

        # High anomaly score (anomalous)
        new_rep = rm.update_reputation("client_1", anomaly_score=0.9)

        # Reputation should decrease
        assert new_rep < 0.5
        assert new_rep >= 0.01  # Bounded

    def test_reputation_bounds(self):
        """Test reputation bounds are enforced."""
        rm = ReputationManager()
        rm.register_client("client_1")

        # Try to push above max
        rm.update_reputation("client_1", 0.0)  # Perfect
        assert rm.get_reputation("client_1") <= 1.0

        # Try to push below min
        for _ in range(10):
            rm.update_reputation("client_1", 1.0)  # Max anomaly
        assert rm.get_reputation("client_1") >= 0.01

    def test_batch_update(self):
        """Test batch reputation update."""
        rm = ReputationManager()

        anomaly_scores = {
            "client_1": 0.1,
            "client_2": 0.5,
            "client_3": 0.9
        }

        updated = rm.batch_update(anomaly_scores)

        # Check all updated
        assert len(updated) == 3
        assert "client_1" in updated
        assert "client_2" in updated
        assert "client_3" in updated

        # client_1 should have highest reputation
        assert updated["client_1"] > updated["client_2"]
        assert updated["client_2"] > updated["client_3"]

    def test_effective_weight_probation(self):
        """Test effective weight during probation."""
        rm = ReputationManager()
        rm.register_client("client_1")

        # Client on probation (0 rounds)
        base_rep = rm.get_reputation("client_1")
        effective = rm.get_effective_weight("client_1")

        # Should be reduced during probation
        assert effective < base_rep
        assert effective == base_rep * rm.probation_weight_multiplier

    def test_effective_weight_after_probation(self):
        """Test effective weight after probation."""
        rm = ReputationManager()
        rm.register_client("client_1")

        # Complete probation rounds
        for _ in range(rm.probation_rounds + 1):
            rm.update_reputation("client_1", 0.1)

        base_rep = rm.get_reputation("client_1")
        effective = rm.get_effective_weight("client_1")

        # Should equal base reputation
        assert effective == base_rep

    def test_reputation_history(self):
        """Test reputation history tracking."""
        rm = ReputationManager()
        rm.register_client("client_1")

        # Update reputation multiple times
        for i in range(5):
            rm.update_reputation("client_1", 0.1 * i)

        history = rm.get_reputation_history("client_1")

        assert len(history) == 5

    def test_reputation_stats(self):
        """Test reputation statistics computation."""
        rm = ReputationManager()

        for i in range(10):
            rm.register_client(f"client_{i}")

        stats = rm.get_reputation_stats()

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats

        assert stats['mean'] == 0.5  # All initial

    def test_get_low_reputation_clients(self):
        """Test filtering low reputation clients."""
        rm = ReputationManager()

        rm.register_client("client_1")
        # Multiple high anomaly updates to drive reputation low
        for _ in range(5):
            rm.update_reputation("client_1", 0.9)  # High anomaly -> low rep

        rm.register_client("client_2")
        rm.update_reputation("client_2", 0.1)  # Low anomaly -> high rep

        low_clients = rm.get_low_reputation_clients(threshold=0.48)

        assert "client_1" in low_clients
        assert "client_2" not in low_clients

    def test_get_high_reputation_clients(self):
        """Test filtering high reputation clients."""
        rm = ReputationManager()

        rm.register_client("client_1")
        rm.update_reputation("client_1", 0.9)  # High anomaly -> low rep

        rm.register_client("client_2")
        # Multiple low anomaly updates to drive reputation high
        for _ in range(5):
            rm.update_reputation("client_2", 0.1)  # Low anomaly -> high rep

        high_clients = rm.get_high_reputation_clients(threshold=0.6)

        assert "client_1" not in high_clients
        assert "client_2" in high_clients

    def test_reset_client(self):
        """Test resetting a single client."""
        rm = ReputationManager()
        rm.register_client("client_1")
        rm.update_reputation("client_1", 0.5)

        rm.reset_client("client_1")

        assert rm.get_reputation("client_1") == 0.5  # Reset to initial
        assert len(rm.get_reputation_history("client_1")) == 0

    def test_reset_all(self):
        """Test resetting all clients."""
        rm = ReputationManager()

        for i in range(5):
            rm.register_client(f"client_{i}")
            rm.update_reputation(f"client_{i}", 0.3)

        rm.reset_all()

        assert len(rm.get_all_reputations()) == 0


class TestWeightedAggregator:
    """Test WeightedAggregator functionality."""

    def test_compute_weights(self):
        """Test weight computation from reputations."""
        rm = ReputationManager()
        rm.register_client("client_1")
        rm.register_client("client_2")
        rm.update_reputation("client_1", 0.1)  # High rep
        rm.update_reputation("client_2", 0.9)  # Low rep

        agg = WeightedAggregator(reputation_manager=rm)
        weights = agg.compute_weights(["client_1", "client_2"])

        # Should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

        # client_1 should have higher weight
        assert weights["client_1"] > weights["client_2"]

    def test_compute_weights_normalized(self):
        """Test weight normalization."""
        rm = ReputationManager()
        rm.register_client("client_1")
        rm.register_client("client_2")
        rm.register_client("client_3")

        agg = WeightedAggregator(reputation_manager=rm)
        weights = agg.compute_weights(["client_1", "client_2", "client_3"])

        # All equal initially, should have uniform weights
        assert abs(weights["client_1"] - 1.0/3.0) < 1e-6
        assert abs(weights["client_2"] - 1.0/3.0) < 1e-6
        assert abs(weights["client_3"] - 1.0/3.0) < 1e-6

    def test_aggregate_updates(self):
        """Test update aggregation."""
        rm = ReputationManager()
        rm.register_client("client_1")
        rm.register_client("client_2")

        # Set different reputations
        rm.update_reputation("client_1", 0.1)  # High rep
        rm.update_reputation("client_2", 0.9)  # Low rep

        agg = WeightedAggregator(reputation_manager=rm)

        # Create updates
        updates = {
            "client_1": [np.array([1.0, 2.0, 3.0])],
            "client_2": [np.array([4.0, 5.0, 6.0])]
        }

        aggregated, metadata = agg.aggregate_updates(updates)

        # Should be weighted average (closer to client_1)
        expected = np.array([
            (4.0 * 0.1 + 1.0 * 0.9) / 1.0,
            (5.0 * 0.1 + 2.0 * 0.9) / 1.0,
            (6.0 * 0.1 + 3.0 * 0.9) / 1.0
        ]) * rm.get_reputation("client_1") / (
            rm.get_reputation("client_1") + rm.get_reputation("client_2")
        ) + np.array([
            (4.0 * 0.9 + 1.0 * 0.1) / 1.0,
            (5.0 * 0.9 + 2.0 * 0.1) / 1.0,
            (6.0 * 0.9 + 3.0 * 0.1) / 1.0
        ]) * rm.get_reputation("client_2") / (
            rm.get_reputation("client_1") + rm.get_reputation("client_2")
        )

        # Just check it's between the two updates
        assert aggregated[0][0] > 1.0 and aggregated[0][0] < 4.0

        # Check metadata
        assert 'weights' in metadata
        assert 'num_clients' in metadata
        assert metadata['num_clients'] == 2

    def test_aggregate_with_num_examples(self):
        """Test aggregation weighted by num_examples."""
        rm = ReputationManager()
        rm.register_client("client_1")
        rm.register_client("client_2")

        agg = WeightedAggregator(reputation_manager=rm)

        updates = {
            "client_1": (np.array([1.0, 2.0]), 100),
            "client_2": (np.array([3.0, 4.0]), 100)
        }

        aggregated, metadata = agg.aggregate_with_num_examples(updates)

        assert 'total_examples' in metadata
        assert metadata['total_examples'] == 200

    def test_can_aggregate(self):
        """Test aggregation eligibility check."""
        rm = ReputationManager()
        agg = WeightedAggregator(reputation_manager=rm)

        # Register clients with good reputations
        rm.register_client("client_1")
        rm.update_reputation("client_1", 0.1)

        can, reason = agg.can_aggregate(["client_1"])

        assert can
        assert reason == "OK"

    def test_cannot_aggregate_low_reputation(self):
        """Test aggregation fails with low total reputation."""
        rm = ReputationManager()
        agg = WeightedAggregator(reputation_manager=rm)

        # Client with very low reputation
        rm.register_client("client_1")
        for _ in range(10):
            rm.update_reputation("client_1", 1.0)

        can, reason = agg.can_aggregate(["client_1"])

        assert not can
        assert "below minimum" in reason

    def test_get_aggregation_stats(self):
        """Test aggregation statistics."""
        rm = ReputationManager()
        rm.register_client("client_1")
        rm.register_client("client_2")

        agg = WeightedAggregator(reputation_manager=rm)
        stats = agg.get_aggregation_stats()

        assert 'num_clients' in stats
        assert 'total_reputation' in stats
        assert 'min_weight' in stats
        assert 'max_weight' in stats
        assert 'weight_entropy' in stats

        assert stats['num_clients'] == 2


class TestStandaloneFunctions:
    """Test standalone utility functions."""

    def test_update_reputation_function(self):
        """Test standalone update_reputation function."""
        new_rep = update_reputation(
            current_reputation=0.5,
            anomaly_score=0.2,
            decay_factor=0.9
        )

        # Should increase (low anomaly)
        assert new_rep > 0.5

    def test_aggregate_updates_function(self):
        """Test standalone aggregate_updates function."""
        updates = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0])
        ]
        weights = [0.3, 0.7]

        aggregated = aggregate_updates(updates, weights)

        expected = 0.3 * np.array([1.0, 2.0, 3.0]) + 0.7 * np.array([4.0, 5.0, 6.0])

        np.testing.assert_array_almost_equal(aggregated, expected)

    def test_compute_weights_function(self):
        """Test standalone compute_weights function."""
        reputations = {
            "client_1": 0.9,
            "client_2": 0.6,
            "client_3": 0.3
        }

        weights = compute_weights(reputations, min_reputation=0.2)

        # Should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

        # client_1 should have highest weight
        assert weights["client_1"] > weights["client_2"]
        assert weights["client_2"] > weights["client_3"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
