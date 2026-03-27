"""
Tests for dropout recovery protocol.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.protocol.dropout_recovery import (
    coordinate_recovery_protocol,
    validate_threshold_sufficient,
    simulate_dropouts,
    analyze_recovery_capability,
    graceful_degradation_analysis
)

from src.protocol.server import SecureAggregationServer
from src.protocol.client import SecureAggregationClient


class TestDropoutRecovery:
    """Test suite for dropout recovery."""

    def test_validate_threshold_sufficient(self):
        """Test threshold validation."""
        # 10 clients, threshold 7
        # Can handle up to 3 dropouts
        assert validate_threshold_sufficient(10, 2, 7)
        assert validate_threshold_sufficient(10, 3, 7)
        assert not validate_threshold_sufficient(10, 4, 7)

    def test_simulate_dropouts(self):
        """Test dropout simulation."""
        client_ids = list(range(10))
        dropout_rate = 0.3

        active, dead = simulate_dropouts(client_ids, dropout_rate, seed=42)

        # All clients should be accounted for
        assert set(active + dead) == set(client_ids)
        assert len(set(active) & set(dead)) == 0  # No overlap

    def test_analyze_recovery_capability(self):
        """Test recovery capability analysis."""
        analysis = analyze_recovery_capability(
            num_clients=10,
            threshold=7,
            max_dropout_rate=0.3
        )

        assert analysis['num_clients'] == 10
        assert analysis['threshold'] == 7
        assert analysis['max_tolerable_dropouts'] == 3
        assert len(analysis['breakdown']) == 5  # 5 rates tested

    def test_graceful_degradation_analysis(self):
        """Test graceful degradation analysis."""
        degradation = graceful_degradation_analysis(
            num_clients=10,
            threshold=7
        )

        # Should have entry for each possible number of dead clients
        assert len(degradation) == 11  # 0 to 10

        # Check specific cases
        assert 'OPTIMAL' in degradation[0]
        assert 'OPERATIONAL' in degradation[2]
        assert 'FAIL' in degradation[4]

    def test_coordinate_recovery_no_dropouts(self):
        """Test recovery protocol with no dropouts."""
        config = {
            'num_clients': 10,
            'threshold_ratio': 0.7,
            'secret_sharing_prime': 2**127 - 1
        }

        model_shape = torch.Size([100])
        server = SecureAggregationServer(10, model_shape, config)

        clients = []
        for i in range(10):
            client = SecureAggregationClient(
                i,
                torch.randn(100),
                config
            )
            clients.append(client)

        # No dropouts
        success = coordinate_recovery_protocol(server, clients, [])

        # Should succeed (nothing to recover)
        assert success

    def test_coordinate_recovery_with_dropouts(self):
        """Test recovery protocol with dropouts."""
        config = {
            'num_clients': 10,
            'threshold_ratio': 0.7,
            'secret_sharing_prime': 2**127 - 1
        }

        model_shape = torch.Size([100])
        server = SecureAggregationServer(10, model_shape, config)

        # Create clients and set up masks
        clients = []
        for i in range(10):
            client = SecureAggregationClient(
                i,
                torch.randn(100),
                config
            )
            client.generate_masks_and_seeds()
            client.create_mask_shares(client.state.my_mask_seed, 10)
            clients.append(client)

        # Simulate 2 dropouts (within threshold)
        active_clients = clients[:8]
        dead_ids = [8, 9]

        success = coordinate_recovery_protocol(server, active_clients, dead_ids)

        # Should succeed with sufficient clients
        assert success

    def test_coordinate_recovery_too_many_dropouts(self):
        """Test recovery protocol with too many dropouts."""
        config = {
            'num_clients': 10,
            'threshold_ratio': 0.7,
            'secret_sharing_prime': 2**127 - 1
        }

        model_shape = torch.Size([100])
        server = SecureAggregationServer(10, model_shape, config)

        # Create clients and set up masks
        clients = []
        for i in range(10):
            client = SecureAggregationClient(
                i,
                torch.randn(100),
                config
            )
            client.generate_masks_and_seeds()
            client.create_mask_shares(client.state.my_mask_seed, 10)
            clients.append(client)

        # Simulate 4 dropouts (exceeds threshold-1=3)
        active_clients = clients[:6]
        dead_ids = [6, 7, 8, 9]

        success = coordinate_recovery_protocol(server, active_clients, dead_ids)

        # Should fail with insufficient clients
        assert not success


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
