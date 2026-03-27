"""
Unit tests for similarity computation functions.

Tests edge cases:
- Empty lists
- Single client
- All clients similar
- All clients different
- Zero vectors
"""

import pytest
import numpy as np
from src.utils.similarity import (
    flatten_parameters,
    cosine_similarity,
    compute_pairwise_cosine_similarity,
    compute_adaptive_weights
)
from flwr.common import ndarrays_to_parameters


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v1, v2) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-1.0, -2.0, -3.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_zero_vectors(self):
        """Zero vectors should return 0.0."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_both_zero_vectors(self):
        """Both zero vectors should return 0.0."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([0.0, 0.0])
        assert cosine_similarity(v1, v2) == 0.0


class TestPairwiseSimilarity:
    """Test pairwise similarity matrix computation."""

    def test_empty_list(self):
        """Empty list should return empty array."""
        result = compute_pairwise_cosine_similarity([])
        assert result.shape == (0,)

    def test_single_client(self):
        """Single client should return 1x1 matrix with value 1.0."""
        gradients = [np.array([1.0, 2.0, 3.0])]
        result = compute_pairwise_cosine_similarity(gradients)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0)

    def test_two_clients_identical(self):
        """Two identical clients should have high similarity."""
        gradients = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0])
        ]
        result = compute_pairwise_cosine_similarity(gradients)
        assert result.shape == (2, 2)
        assert result[0, 1] == pytest.approx(1.0)
        assert result[1, 0] == pytest.approx(1.0)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[1, 1] == pytest.approx(1.0)

    def test_two_clients_opposite(self):
        """Two opposite clients should have negative similarity."""
        gradients = [
            np.array([1.0, 2.0, 3.0]),
            np.array([-1.0, -2.0, -3.0])
        ]
        result = compute_pairwise_cosine_similarity(gradients)
        assert result[0, 1] == pytest.approx(-1.0)
        assert result[1, 0] == pytest.approx(-1.0)

    def test_all_clients_similar_sybil_scenario(self):
        """All clients with similar updates (Sybil attack)."""
        base = np.array([1.0, 2.0, 3.0])
        noise = 0.01
        gradients = [
            base + np.random.randn(3) * noise,
            base + np.random.randn(3) * noise,
            base + np.random.randn(3) * noise,
            base + np.random.randn(3) * noise,
        ]
        result = compute_pairwise_cosine_similarity(gradients)
        assert result.shape == (4, 4)
        # All off-diagonal should be high (> 0.9)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert result[i, j] > 0.9

    def test_all_clients_different(self):
        """All clients with different updates."""
        gradients = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
        ]
        result = compute_pairwise_cosine_similarity(gradients)
        assert result.shape == (4, 4)
        # Off-diagonal should be relatively low
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert result[i, j] < 0.8

    def test_symmetry(self):
        """Similarity matrix should be symmetric."""
        gradients = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
        ]
        result = compute_pairwise_cosine_similarity(gradients)
        assert np.allclose(result, result.T)


class TestAdaptiveWeights:
    """Test adaptive weight computation."""

    def test_empty_similarity(self):
        """Empty similarity matrix should return empty weights."""
        result = compute_adaptive_weights(np.array([]))
        assert result.shape == (0,)

    def test_single_client(self):
        """Single client should get weight 1.0."""
        result = compute_adaptive_weights(np.array([[1.0]]))
        assert result.shape == (1,)
        assert result[0] == pytest.approx(1.0)

    def test_high_similarity_gets_lower_weight(self):
        """Clients with high similarity should get lower weights."""
        # All clients very similar
        sim_matrix = np.array([
            [1.0, 0.95, 0.95],
            [0.95, 1.0, 0.95],
            [0.95, 0.95, 1.0]
        ])
        weights = compute_adaptive_weights(sim_matrix, lr_scale_factor=0.5)
        assert len(weights) == 3
        # All should have similar, reduced weights
        assert all(w < 1.0 for w in weights)

    def test_low_similarity_gets_higher_weight(self):
        """Clients with low similarity should get higher weights."""
        # All clients very different
        sim_matrix = np.array([
            [1.0, 0.1, 0.1],
            [0.1, 1.0, 0.1],
            [0.1, 0.1, 1.0]
        ])
        weights = compute_adaptive_weights(sim_matrix, lr_scale_factor=0.5)
        assert len(weights) == 3
        # All should have higher weights
        assert all(w > 0.8 for w in weights)

    def test_weights_sum_to_num_clients(self):
        """Weights should sum to num_clients (preserve scale)."""
        sim_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        weights = compute_adaptive_weights(sim_matrix)
        assert np.sum(weights) == pytest.approx(3.0)


class TestFlattenParameters:
    """Test parameter flattening."""

    def test_flatten_single_layer(self):
        """Flatten single layer parameters."""
        arrays = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        params = ndarrays_to_parameters(arrays)
        result = flatten_parameters(params)
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.array_equal(result, expected)

    def test_flatten_multiple_layers(self):
        """Flatten multi-layer parameters."""
        arrays = [
            np.array([[1.0, 2.0]]),
            np.array([[3.0], [4.0]]),
            np.array([5.0])
        ]
        params = ndarrays_to_parameters(arrays)
        result = flatten_parameters(params)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
