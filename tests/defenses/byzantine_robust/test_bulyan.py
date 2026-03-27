"""
Unit tests for Bulyan aggregator.
"""

import pytest
import torch
from src.aggregators.bulyan import Bulyan, bulyan


class TestBulyan:
    """Test cases for Bulyan aggregation."""

    def test_bulyan_basic(self):
        """Test basic Bulyan aggregation."""
        # n=9, f=2 (requires n >= 4f + 1 = 9, which is satisfied)
        updates = [
            {'weight': torch.tensor([10.0])},  # Honest
            {'weight': torch.tensor([11.0])},  # Honest
            {'weight': torch.tensor([12.0])},  # Honest
            {'weight': torch.tensor([13.0])},  # Honest
            {'weight': torch.tensor([14.0])},  # Honest
            {'weight': torch.tensor([-100.0])},  # Attacker 1
            {'weight': torch.tensor([-200.0])},  # Attacker 2
            {'weight': torch.tensor([200.0])},   # Random
            {'weight': torch.tensor([300.0])},   # Random
        ]

        aggregator = Bulyan()
        result = aggregator.aggregate(updates, num_attackers=2)

        # Bulyan should filter out attackers and return result close to honest mean
        # Honest values: [10, 11, 12, 13, 14]
        # Result should be close to 12 (median)
        assert result['weight'].item() > 5 and result['weight'].item() < 20

    def test_bulyan_robustness_constraints(self):
        """Test that Bulyan enforces n >= 4f + 1."""
        updates = [
            {'weight': torch.tensor([float(i)])} for i in range(10)
        ]

        aggregator = Bulyan()

        # n=10, max_f = (10-1)//4 = 2
        # f=3 should fail (requires n >= 13)
        with pytest.raises(ValueError, match="requires n >= 4f \\+ 1"):
            aggregator.aggregate(updates, num_attackers=3)

        # f=2 should work (requires n >= 9, we have 10)
        result = aggregator.aggregate(updates, num_attackers=2)
        assert 'weight' in result

    def test_bulyan_with_multiple_parameters(self):
        """Test Bulyan with models having multiple parameters."""
        updates = [
            {
                'layer1.weight': torch.tensor([[10.0]]),
                'layer2.bias': torch.tensor([20.0])
            },
            {
                'layer1.weight': torch.tensor([[11.0]]),
                'layer2.bias': torch.tensor([21.0])
            },
            {
                'layer1.weight': torch.tensor([[12.0]]),
                'layer2.bias': torch.tensor([22.0])
            },
            {
                'layer1.weight': torch.tensor([[13.0]]),
                'layer2.bias': torch.tensor([23.0])
            },
            {
                'layer1.weight': torch.tensor([[-100.0]]),  # Attacker
                'layer2.bias': torch.tensor([-200.0])
            },
            {
                'layer1.weight': torch.tensor([[200.0]]),  # Attacker
                'layer2.bias': torch.tensor([400.0])
            },
            {
                'layer1.weight': torch.tensor([[14.0]]),
                'layer2.bias': torch.tensor([24.0])
            },
            {
                'layer1.weight': torch.tensor([[15.0]]),
                'layer2.bias': torch.tensor([25.0])
            },
            {
                'layer1.weight': torch.tensor([[16.0]]),
                'layer2.bias': torch.tensor([26.0])
            },
        ]

        aggregator = Bulyan()
        result = aggregator.aggregate(updates, num_attackers=2)

        # Should have both parameters
        assert 'layer1.weight' in result
        assert 'layer2.bias' in result

        # Should be close to honest values (not affected by attackers)
        # Honest means: ~12.5 and ~22.5
        assert 10 < result['layer1.weight'].item() < 15
        assert 20 < result['layer2.bias'].item() < 25

    def test_bulyan_minimal_configuration(self):
        """Test Bulyan with minimal n=4f+1 configuration."""
        # n=9, f=2 is minimal
        updates = [
            {'weight': torch.tensor([10.0])},
            {'weight': torch.tensor([11.0])},
            {'weight': torch.tensor([12.0])},
            {'weight': torch.tensor([13.0])},
            {'weight': torch.tensor([14.0])},
            {'weight': torch.tensor([15.0])},
            {'weight': torch.tensor([16.0])},
            {'weight': torch.tensor([100.0])},  # Attacker
            {'weight': torch.tensor([-100.0])},  # Attacker
        ]

        aggregator = Bulyan()
        result = aggregator.aggregate(updates, num_attackers=2)

        # Should still produce reasonable result
        assert 5 < result['weight'].item() < 25

    def test_bulyan_selection_step(self):
        """Test that Krum selection step works correctly."""
        # Create updates where specific ones are more central
        updates = []
        for i in range(9):
            if i < 5:
                # Honest: clustered around 10
                updates.append({'weight': torch.tensor([10.0 + i * 0.1])})
            else:
                # Attackers: far away
                updates.append({'weight': torch.tensor([100.0 + i * 10.0])})

        aggregator = Bulyan()
        result = aggregator.aggregate(updates, num_attackers=2)

        # Result should be close to honest cluster
        assert result['weight'].item() < 20

    def test_functional_interface(self):
        """Test bulyan() function."""
        updates = [
            {'weight': torch.tensor([float(i)])} for i in range(10)
        ]

        result = bulyan(updates, num_attackers=2)
        assert 'weight' in result

    def test_repr(self):
        """Test string representation."""
        aggregator = Bulyan()
        assert repr(aggregator) == "Bulyan()"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
