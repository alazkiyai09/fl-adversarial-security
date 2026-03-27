"""
Unit tests for Coordinate-wise Median aggregator.
"""

import pytest
import torch
from src.aggregators.median import CoordinateWiseMedian, coordinate_wise_median


class TestCoordinateWiseMedian:
    """Test cases for coordinate-wise median aggregation."""

    def test_simple_median_1d(self):
        """Test median computation with 1D tensors."""
        updates = [
            {'weight': torch.tensor([1.0, 2.0, 3.0])},
            {'weight': torch.tensor([4.0, 5.0, 6.0])},
            {'weight': torch.tensor([7.0, 8.0, 9.0])},
        ]

        aggregator = CoordinateWiseMedian()
        result = aggregator.aggregate(updates, num_attackers=1)

        # Median of [1,4,7], [2,5,8], [3,6,9] should be [4,5,6]
        expected = torch.tensor([4.0, 5.0, 6.0])
        assert torch.allclose(result['weight'], expected)

    def test_median_with_even_number(self):
        """Test median with even number of clients."""
        updates = [
            {'weight': torch.tensor([1.0])},
            {'weight': torch.tensor([2.0])},
            {'weight': torch.tensor([3.0])},
            {'weight': torch.tensor([4.0])},
        ]

        aggregator = CoordinateWiseMedian()
        result = aggregator.aggregate(updates, num_attackers=1)

        # torch.median returns the lower of two middle values
        # Median of [1,2,3,4] = 2 (lower middle), not 2.5
        expected = torch.tensor([2.0])
        assert torch.allclose(result['weight'], expected)

    def test_median_robustness_to_outliers(self):
        """Test that median is robust to extreme outliers."""
        # One attacker with huge values
        updates = [
            {'weight': torch.tensor([1.0, 1.0])},
            {'weight': torch.tensor([2.0, 2.0])},
            {'weight': torch.tensor([3.0, 3.0])},
            {'weight': torch.tensor([1000.0, 1000.0])},  # Malicious
        ]

        aggregator = CoordinateWiseMedian()
        result = aggregator.aggregate(updates, num_attackers=1)

        # torch.median returns lower middle value
        # Median of [1,2,3,1000] = 2 (lower middle), not 2.5
        expected = torch.tensor([2.0, 2.0])
        assert torch.allclose(result['weight'], expected)

    def test_median_multi_dimensional(self):
        """Test median with multi-dimensional tensors."""
        updates = [
            {'layer.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
            {'layer.weight': torch.tensor([[5.0, 6.0], [7.0, 8.0]])},
            {'layer.weight': torch.tensor([[9.0, 10.0], [11.0, 12.0]])},
        ]

        aggregator = CoordinateWiseMedian()
        result = aggregator.aggregate(updates, num_attackers=1)

        # Element-wise median
        expected = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        assert torch.allclose(result['layer.weight'], expected)

    def test_empty_updates_raises_error(self):
        """Test that empty updates list raises ValueError."""
        aggregator = CoordinateWiseMedian()
        with pytest.raises(ValueError, match="Cannot aggregate empty list"):
            aggregator.aggregate([], num_attackers=1)

    def test_insufficient_clients_for_robustness(self):
        """Test that n > 2f is enforced."""
        updates = [
            {'weight': torch.tensor([1.0])},
            {'weight': torch.tensor([2.0])},
        ]

        aggregator = CoordinateWiseMedian()
        # n=2, f=1 violates n > 2f (2 > 2 is false)
        with pytest.raises(ValueError, match="requires n > 2f"):
            aggregator.aggregate(updates, num_attackers=1)

    def test_mismatched_structure_raises_error(self):
        """Test that mismatched update structures raise ValueError."""
        updates = [
            {'weight': torch.tensor([1.0])},
            {'bias': torch.tensor([2.0])},  # Different key
        ]

        aggregator = CoordinateWiseMedian()
        with pytest.raises(ValueError, match="mismatched keys"):
            aggregator.aggregate(updates, num_attackers=1)

    def test_functional_interface(self):
        """Test the functional interface."""
        updates = [
            {'weight': torch.tensor([1.0])},
            {'weight': torch.tensor([2.0])},
            {'weight': torch.tensor([3.0])},
        ]

        result = coordinate_wise_median(updates, num_attackers=1)
        expected = torch.tensor([2.0])
        assert torch.allclose(result['weight'], expected)

    def test_repr(self):
        """Test string representation."""
        aggregator = CoordinateWiseMedian()
        assert repr(aggregator) == "CoordinateWiseMedian()"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
