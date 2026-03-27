"""
Unit tests for Krum and Multi-Krum aggregators.
"""

import pytest
import torch
from src.aggregators.krum import Krum, MultiKrum, krum, multi_krum
from src.utils.geometry import flatten_update


class TestKrum:
    """Test cases for Krum aggregation."""

    def test_krum_selects_central_update(self):
        """Test that Krum selects the most central update."""
        # Create 5 updates where update 2 is most central
        # Updates 0 and 4 are far outliers, 1 and 3 are moderately far
        updates = [
            {'weight': torch.tensor([0.0, 0.0])},  # Far outlier
            {'weight': torch.tensor([8.0, 8.0])},  # Moderate
            {'weight': torch.tensor([10.0, 10.0])},  # Central (target)
            {'weight': torch.tensor([12.0, 12.0])},  # Moderate
            {'weight': torch.tensor([20.0, 20.0])},  # Far outlier
        ]

        aggregator = Krum()
        result = aggregator.aggregate(updates, num_attackers=1)

        # Update 2 should be selected (most central)
        expected = torch.tensor([10.0, 10.0])
        assert torch.allclose(result['weight'], expected)

    def test_krum_with_tie(self):
        """Test Krum behavior when multiple updates have equal scores."""
        updates = [
            {'weight': torch.tensor([0.0])},
            {'weight': torch.tensor([10.0])},
            {'weight': torch.tensor([20.0])},
        ]

        aggregator = Krum()
        result = aggregator.aggregate(updates, num_attackers=0)

        # With f=0, num_closest = 3 - 0 - 2 = 1
        # Each update has sum of 1 closest distance
        # Should return one of the updates (implementation-dependent)
        # Just verify it has the correct structure
        assert 'weight' in result
        assert result['weight'].shape == torch.tensor([0.0]).shape

    def test_krum_robustness_constraints(self):
        """Test that Krum enforces n >= 3f + 3."""
        updates = [
            {'weight': torch.tensor([float(i)])} for i in range(10)
        ]

        aggregator = Krum()

        # n=10, max_f = (10-2)//3 = 2
        # f=3 should fail
        with pytest.raises(ValueError, match="requires n >= 3f \\+ 3"):
            aggregator.aggregate(updates, num_attackers=3)

        # f=2 should work
        result = aggregator.aggregate(updates, num_attackers=2)
        assert 'weight' in result

    def test_krum_score_computation(self):
        """Test Krum score calculation."""
        n = 5
        f = 1

        # Create distance matrix where:
        # - Update 0 is close to 1 and 2
        # - Update 1 is close to 0 and 2
        # - Update 2 is close to 0, 1, 3, 4 (most central)
        # - Updates 3, 4 are farther away

        flattened = [
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([10.0]),
            torch.tensor([20.0]),
        ]

        from src.utils.geometry import pairwise_distances
        distances = pairwise_distances(flattened)

        aggregator = Krum()
        scores = aggregator._compute_krum_scores(distances, f, n)

        # num_closest = n - f - 2 = 5 - 1 - 2 = 2
        # Update 1 (value=1) is closest to 0 and 2 (distances 1, 1) = score 2
        # Update 0 (value=0) is closest to 1 (dist 1), then 2 (dist 2) = score 3
        # Update 2 (value=2) is closest to 1 (dist 1), then 0 (dist 2) = score 3
        # So update 1 should have lowest score
        assert torch.argmin(scores) == 1

    def test_krum_with_multiple_parameters(self):
        """Test Krum with models having multiple parameters."""
        updates = [
            {'layer1.weight': torch.tensor([[1.0]]), 'layer2.bias': torch.tensor([2.0])},
            {'layer1.weight': torch.tensor([[10.0]]), 'layer2.bias': torch.tensor([20.0])},
            {'layer1.weight': torch.tensor([[3.0]]), 'layer2.bias': torch.tensor([4.0])},
        ]

        aggregator = Krum()
        result = aggregator.aggregate(updates, num_attackers=0)

        # Should select the most central update
        assert 'layer1.weight' in result
        assert 'layer2.bias' in result


class TestMultiKrum:
    """Test cases for Multi-Krum aggregation."""

    def test_multi_krum_averages_multiple_updates(self):
        """Test that Multi-Krum averages m selected updates."""
        updates = [
            {'weight': torch.tensor([10.0])},  # Central
            {'weight': torch.tensor([11.0])},  # Central
            {'weight': torch.tensor([12.0])},  # Central
            {'weight': torch.tensor([0.0])},   # Outlier
            {'weight': torch.tensor([100.0])},  # Outlier
        ]

        aggregator = MultiKrum(m=3)
        result = aggregator.aggregate(updates, num_attackers=1)

        # Should average the 3 central updates: (10 + 11 + 12) / 3 = 11
        expected = torch.tensor([11.0])
        assert torch.allclose(result['weight'], expected)

    def test_multi_krum_auto_m(self):
        """Test Multi-Krum with auto-computed m."""
        updates = [
            {'weight': torch.tensor([float(i)])} for i in range(10)
        ]

        aggregator = MultiKrum(m=None)  # Auto-compute
        result = aggregator.aggregate(updates, num_attackers=2)

        # With n=10, f=2: m = n - f - 2 = 10 - 2 - 2 = 6
        # Should average 6 selected updates
        assert 'weight' in result

    def test_multi_krum_m_exceeds_n(self):
        """Test error when m exceeds number of updates."""
        updates = [
            {'weight': torch.tensor([1.0])},
            {'weight': torch.tensor([2.0])},
        ]

        aggregator = MultiKrum(m=5)
        with pytest.raises(ValueError, match="m=.*cannot be larger than n="):
            aggregator.aggregate(updates, num_attackers=0)

    def test_multi_krum_m_too_small(self):
        """Test error when f exceeds robustness threshold before m check."""
        updates = [
            {'weight': torch.tensor([1.0])} for _ in range(10)
        ]

        aggregator = MultiKrum(m=2)
        # Multi-Krum first checks if f is within robustness threshold
        # n=10, max_f = 2, so f=3 fails the robustness check before m check
        with pytest.raises(ValueError, match="requires n >= 3f"):
            aggregator.aggregate(updates, num_attackers=3)

    def test_multi_krum_robustness_constraints(self):
        """Test that Multi-Krum enforces n >= 3f + 3."""
        updates = [
            {'weight': torch.tensor([float(i)])} for i in range(10)
        ]

        aggregator = MultiKrum(m=5)

        # n=10, max_f = 2, f=3 should fail
        with pytest.raises(ValueError, match="requires n >= 3f \\+ 3"):
            aggregator.aggregate(updates, num_attackers=3)

    def test_multi_krum_with_different_m(self):
        """Test Multi-Krum with different m values."""
        updates = [
            {'weight': torch.tensor([float(i)])} for i in range(10)
        ]

        for m in [3, 5, 7]:
            aggregator = MultiKrum(m=m)
            result = aggregator.aggregate(updates, num_attackers=2)
            assert 'weight' in result


class TestFunctionalInterfaces:
    """Test functional interfaces for Krum and Multi-Krum."""

    def test_krum_functional(self):
        """Test krum() function."""
        updates = [
            {'weight': torch.tensor([1.0])},
            {'weight': torch.tensor([2.0])},
            {'weight': torch.tensor([3.0])},
        ]

        result = krum(updates, num_attackers=0)
        assert 'weight' in result

    def test_multi_krum_functional(self):
        """Test multi_krum() function."""
        updates = [
            {'weight': torch.tensor([float(i)])} for i in range(10)
        ]

        result = multi_krum(updates, num_attackers=2, m=5)
        assert 'weight' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
