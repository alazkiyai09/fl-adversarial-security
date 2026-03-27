"""
Unit tests for FoolsGold aggregator.

Tests algorithm correctness, edge cases, and integration.
"""

import pytest
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters

from src.aggregators.foolsgold import (
    FoolsGoldAggregator,
    compute_contribution_scores,
    compute_pairwise_cosine_similarity,
    foolsgold_aggregate
)


class TestPairwiseSimilarity:
    """Test pairwise cosine similarity computation."""

    def test_empty_gradients(self):
        """Empty list should return empty array."""
        result = compute_pairwise_cosine_similarity([])
        assert result.shape == (0,)

    def test_single_gradient(self):
        """Single gradient should return 1x1 matrix with value 1.0."""
        gradients = [np.array([1.0, 2.0, 3.0])]
        result = compute_pairwise_cosine_similarity(gradients)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0)

    def test_identical_gradients(self):
        """Identical gradients should have similarity 1.0."""
        gradients = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0])
        ]
        result = compute_pairwise_cosine_similarity(gradients)
        assert result[0, 1] == pytest.approx(1.0)
        assert result[1, 0] == pytest.approx(1.0)

    def test_opposite_gradients(self):
        """Opposite gradients should have similarity -1.0."""
        gradients = [
            np.array([1.0, 2.0, 3.0]),
            np.array([-1.0, -2.0, -3.0])
        ]
        result = compute_pairwise_cosine_similarity(gradients)
        assert result[0, 1] == pytest.approx(-1.0)

    def test_sybil_scenario(self):
        """Sybil attack: malicious clients have high similarity."""
        base = np.array([1.0, 2.0, 3.0])
        noise = 0.01
        gradients = [
            np.array([1.0, 0.0, 0.0]),  # Honest 1
            np.array([0.0, 1.0, 0.0]),  # Honest 2
            base + np.random.randn(3) * noise,  # Sybil 1
            base + np.random.randn(3) * noise,  # Sybil 2
        ]
        result = compute_pairwise_cosine_similarity(gradients)
        # Sybils should have high similarity
        assert result[2, 3] > 0.9


class TestContributionScores:
    """Test contribution score computation."""

    def test_empty_client_list(self):
        """Empty client list should return empty array."""
        result = compute_contribution_scores(
            np.array([]),
            {},
            10,
            []
        )
        assert result.shape == (0,)

    def test_single_client(self):
        """Single client should get score 1.0."""
        result = compute_contribution_scores(
            np.array([[1.0]]),
            {},
            10,
            [0]
        )
        assert result.shape == (1,)
        assert result[0] == pytest.approx(1.0)

    def test_all_dissimilar_clients(self):
        """Dissimilar clients should get similar scores."""
        similarity_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        result = compute_contribution_scores(
            similarity_matrix,
            {},
            10,
            [0, 1, 2]
        )
        assert len(result) == 3
        # All should have similar, relatively high scores
        assert all(s > 0.5 for s in result)

    def test_all_similar_clients(self):
        """Similar clients (Sybils) should get lower scores."""
        similarity_matrix = np.array([
            [1.0, 0.95, 0.95],
            [0.95, 1.0, 0.95],
            [0.95, 0.95, 1.0]
        ])
        result = compute_contribution_scores(
            similarity_matrix,
            {},
            10,
            [0, 1, 2]
        )
        assert len(result) == 3
        # All should have reduced scores due to high similarity
        assert all(s < 1.0 for s in result)

    def test_mixed_similarity(self):
        """Mixed: some clients similar, some different."""
        # Clients 0 and 1 are similar (Sybils)
        # Client 2 is different (honest)
        similarity_matrix = np.array([
            [1.0, 0.95, 0.1],
            [0.95, 1.0, 0.1],
            [0.1, 0.1, 1.0]
        ])
        result = compute_contribution_scores(
            similarity_matrix,
            {},
            10,
            [0, 1, 2]
        )
        assert len(result) == 3
        # Client 2 (honest, dissimilar) should have higher score
        assert result[2] > result[0]
        assert result[2] > result[1]


class TestFoolsGoldAggregate:
    """Test FoolsGold weighted aggregation."""

    def test_empty_parameters(self):
        """Empty parameter list should raise error."""
        with pytest.raises(ValueError):
            foolsgold_aggregate([], np.array([1.0]))

    def test_single_parameter(self):
        """Single parameter should return unchanged."""
        params = [ndarrays_to_parameters([np.array([[1.0, 2.0]])])]
        scores = np.array([1.0])
        result = foolsgold_aggregate(params, scores)

        result_arrays = result.tensors
        assert len(result_arrays) == 1
        assert np.allclose(result_arrays[0], np.array([[1.0, 2.0]]))

    def test_weighted_aggregation(self):
        """Weighted aggregation should weight by scores."""
        params = [
            ndarrays_to_parameters([np.array([1.0])]),
            ndarrays_to_parameters([np.array([2.0])]),
        ]
        scores = np.array([0.7, 0.3])

        result = foolsgold_aggregate(params, scores, lr_scale_factor=0.0)
        result_arrays = result.tensors

        # Should be weighted average
        expected = 0.7 * 1.0 + 0.3 * 2.0
        assert np.allclose(result_arrays[0], expected, atol=0.1)


class TestFoolsGoldAggregator:
    """Test FoolsGold aggregator class."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = FoolsGoldAggregator(
            history_length=10,
            similarity_threshold=0.9,
            lr_scale_factor=0.1
        )

        assert aggregator.history_length == 10
        assert aggregator.similarity_threshold == 0.9
        assert aggregator.lr_scale_factor == 0.1
        assert len(aggregator.gradient_history) == 0

    def test_reset_history(self):
        """Test history reset."""
        aggregator = FoolsGoldAggregator()
        aggregator.gradient_history[0] = [np.array([1.0])]

        aggregator.reset_history()

        assert len(aggregator.gradient_history) == 0
        assert len(aggregator.history["similarity_matrices"]) == 0

    def test_aggregate_empty_results(self):
        """Aggregating empty results should raise error."""
        aggregator = FoolsGoldAggregator()

        with pytest.raises(ValueError):
            aggregator.aggregate([])

    def test_get_metrics(self):
        """Test metrics retrieval."""
        aggregator = FoolsGoldAggregator()
        metrics = aggregator.get_metrics()

        assert "similarity_matrices" in metrics
        assert "contribution_scores" in metrics
        assert "flagged_sybils" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
