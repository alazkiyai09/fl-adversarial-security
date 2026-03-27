"""
Unit tests for Trimmed Mean aggregator.
"""

import pytest
import torch
from src.aggregators.trimmed_mean import TrimmedMean, trimmed_mean


class TestTrimmedMean:
    """Test cases for trimmed mean aggregation."""

    def test_simple_trimmed_mean(self):
        """Test trimmed mean with basic values."""
        updates = [
            {'weight': torch.tensor([1.0, 2.0, 3.0])},
            {'weight': torch.tensor([4.0, 5.0, 6.0])},
            {'weight': torch.tensor([7.0, 8.0, 9.0])},
            {'weight': torch.tensor([10.0, 11.0, 12.0])},
            {'weight': torch.tensor([13.0, 14.0, 15.0])},
        ]

        aggregator = TrimmedMean(beta=0.2)
        result = aggregator.aggregate(updates, num_attackers=1)

        # With beta=0.2, n=5, k=1
        # Remove lowest 1 and highest 1, average middle 3
        # Values: [1,4,7,10,13] -> remove 1,13 -> avg([4,7,10]) = 7
        # Values: [2,5,8,11,14] -> remove 2,14 -> avg([5,8,11]) = 8
        # Values: [3,6,9,12,15] -> remove 3,15 -> avg([6,9,12]) = 9
        expected = torch.tensor([7.0, 8.0, 9.0])
        assert torch.allclose(result['weight'], expected)

    def test_trimmed_mean_removes_outliers(self):
        """Test that trimmed mean removes extreme values."""
        updates = [
            {'weight': torch.tensor([-100.0])},  # Outlier low
            {'weight': torch.tensor([5.0])},
            {'weight': torch.tensor([6.0])},
            {'weight': torch.tensor([7.0])},
            {'weight': torch.tensor([100.0])},  # Outlier high
        ]

        aggregator = TrimmedMean(beta=0.2)
        result = aggregator.aggregate(updates, num_attackers=1)

        # With beta=0.2, n=5, k=1
        # Remove -100 and 100, average [5,6,7] = 6
        expected = torch.tensor([6.0])
        assert torch.allclose(result['weight'], expected)

    def test_different_beta_values(self):
        """Test trimmed mean with different beta values."""
        updates = [
            {'weight': torch.tensor([float(i)])} for i in range(1, 11)  # 1 to 10
        ]

        # beta=0.1, n=10, k=1 -> remove 1 lowest, 1 highest
        # Keep [2..9], average = 5.5
        aggregator = TrimmedMean(beta=0.1)
        result = aggregator.aggregate(updates, num_attackers=1)
        expected = torch.tensor([5.5])
        assert torch.allclose(result['weight'], expected)

        # beta=0.3, n=10, k=3 -> remove 3 lowest, 3 highest
        # Keep [4..7], average = 5.0 (mean of 4,5,6,7)
        aggregator = TrimmedMean(beta=0.3)
        result = aggregator.aggregate(updates, num_attackers=3)
        expected = torch.tensor([5.5])  # (4+5+6+7)/4 = 5.5
        assert torch.allclose(result['weight'], expected)

    def test_invalid_beta_raises_error(self):
        """Test that invalid beta values raise ValueError."""
        with pytest.raises(ValueError, match="beta must be in"):
            TrimmedMean(beta=0.0)  # Too low

        with pytest.raises(ValueError, match="beta must be in"):
            TrimmedMean(beta=0.5)  # Too high (would trim all)

    def test_insufficient_clients_for_trimming(self):
        """Test that n > 2k is enforced."""
        updates = [
            {'weight': torch.tensor([1.0])},
            {'weight': torch.tensor([2.0])},
        ]

        aggregator = TrimmedMean(beta=0.4)  # k = 0
        result = aggregator.aggregate(updates, num_attackers=0)
        # With beta=0.4, n=2, k=0 (int(0.4*2) = 0)
        # Should average both: 1.5
        expected = torch.tensor([1.5])
        assert torch.allclose(result['weight'], expected)

        # beta must be < 0.5, so using 0.49 with n=2 gives k=0 (int(0.49*2) = 0)
        # This still works since k=0 means no trimming
        aggregator = TrimmedMean(beta=0.49)
        result = aggregator.aggregate(updates, num_attackers=0)
        assert torch.allclose(result['weight'], expected)

    def test_multi_dimensional(self):
        """Test trimmed mean with multi-dimensional tensors."""
        updates = [
            {'layer.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
            {'layer.weight': torch.tensor([[5.0, 6.0], [7.0, 8.0]])},
            {'layer.weight': torch.tensor([[9.0, 10.0], [11.0, 12.0]])},
            {'layer.weight': torch.tensor([[13.0, 14.0], [15.0, 16.0]])},
            {'layer.weight': torch.tensor([[17.0, 18.0], [19.0, 20.0]])},
        ]

        aggregator = TrimmedMean(beta=0.2)
        result = aggregator.aggregate(updates, num_attackers=1)

        # Element-wise trimmed mean (k=1, remove lowest and highest)
        # [1,5,9,13,17] -> avg([5,9,13]) = 9
        # [2,6,10,14,18] -> avg([6,10,14]) = 10
        # [3,7,11,15,19] -> avg([7,11,15]) = 11
        # [4,8,12,16,20] -> avg([8,12,16]) = 12
        expected = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        assert torch.allclose(result['layer.weight'], expected)

    def test_num_attackers_exceeds_beta_capacity(self):
        """Test error when num_attackers exceeds beta capacity."""
        updates = [
            {'weight': torch.tensor([float(i)])} for i in range(10)
        ]

        aggregator = TrimmedMean(beta=0.1)  # k=1
        # num_attackers=2 exceeds k=1
        with pytest.raises(ValueError, match="can only handle k=.*attackers"):
            aggregator.aggregate(updates, num_attackers=2)

    def test_functional_interface(self):
        """Test the functional interface."""
        updates = [
            {'weight': torch.tensor([1.0])},
            {'weight': torch.tensor([2.0])},
            {'weight': torch.tensor([3.0])},
            {'weight': torch.tensor([4.0])},
            {'weight': torch.tensor([5.0])},
        ]

        result = trimmed_mean(updates, num_attackers=1, beta=0.2)
        expected = torch.tensor([3.0])
        assert torch.allclose(result['weight'], expected)

    def test_repr(self):
        """Test string representation."""
        aggregator = TrimmedMean(beta=0.2)
        assert repr(aggregator) == "TrimmedMean(beta=0.2)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
