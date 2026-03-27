"""
Integration tests for the full secure aggregation protocol.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.simulation.simplified import run_simplified_simulation


class TestProtocolIntegration:
    """Integration tests for the complete protocol."""

    def test_full_protocol_no_dropouts(self):
        """Test full protocol with no dropouts."""
        result = run_simplified_simulation(
            num_clients=10,
            model_size=100,
            dropout_rate=0.0,
            seed=42
        )

        assert result['success'] == True
        assert result['aggregate_matches'] == True
        assert result['difference'] < 1e-5

    def test_full_protocol_with_dropouts(self):
        """Test full protocol with dropouts."""
        result = run_simplified_simulation(
            num_clients=10,
            model_size=100,
            dropout_rate=0.2,
            seed=42
        )

        # Should still succeed with 20% dropout
        assert result['success'] == True
        assert result['aggregate_matches'] == True

    def test_full_protocol_max_dropouts(self):
        """Test full protocol with maximum tolerable dropouts."""
        result = run_simplified_simulation(
            num_clients=10,
            model_size=100,
            dropout_rate=0.3,
            seed=42
        )

        # Should succeed at 30% dropout
        assert result['success'] == True

    def test_full_protocol_excessive_dropouts(self):
        """Test that excessive dropouts cause failure."""
        result = run_simplified_simulation(
            num_clients=10,
            model_size=100,
            dropout_rate=0.4,
            seed=42
        )

        # Should fail with 40% dropout (>30% tolerance)
        assert result['success'] == False

    def test_different_client_counts(self):
        """Test protocol with different numbers of clients."""
        for num_clients in [5, 10, 20, 50]:
            result = run_simplified_simulation(
                num_clients=num_clients,
                model_size=100,
                dropout_rate=0.0,
                seed=42
            )

            assert result['success'] == True
            assert result['aggregate_matches'] == True

    def test_different_model_sizes(self):
        """Test protocol with different model sizes."""
        for model_size in [10, 100, 1000, 10000]:
            result = run_simplified_simulation(
                num_clients=10,
                model_size=model_size,
                dropout_rate=0.0,
                seed=42
            )

            assert result['success'] == True
            assert result['difference'] < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
