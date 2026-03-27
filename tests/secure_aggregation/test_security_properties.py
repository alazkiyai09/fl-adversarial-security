"""
Tests for security properties of secure aggregation.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.security.verification import (
    verify_server_privacy,
    verify_collusion_resistance,
    verify_forward_secrecy,
    SecurityAuditor
)

from src.crypto.prf import generate_mask_from_seed


class TestSecurityProperties:
    """Test suite for security properties."""

    def test_server_privacy(self):
        """Test that server cannot see individual updates."""
        assert verify_server_privacy(num_clients=10)

    def test_collusion_resistance(self):
        """Test that collusion resistance works."""
        # 10 clients, threshold 7
        # 6 colluding clients should not be able to reconstruct
        assert verify_collusion_resistance(num_clients=10, threshold=7)

    def test_forward_secrecy(self):
        """Test forward secrecy property."""
        assert verify_forward_secrecy()

    def test_mask_unpredictability(self):
        """Test that masks are unpredictable."""
        # Generate masks from different seeds
        mask1 = generate_mask_from_seed(12345, (1000,), torch.float32)
        mask2 = generate_mask_from_seed(12346, (1000,), torch.float32)

        # Masks should be different
        assert not torch.allclose(mask1, mask2)

        # Correlation should be low
        corr = torch.corrcoef(torch.stack([mask1, mask2]))[0, 1].item()
        assert abs(corr) < 0.1

    def test_full_security_audit(self):
        """Test comprehensive security audit."""
        auditor = SecurityAuditor({})

        result = auditor.audit_all_properties()

        assert result['total_tests'] == 4  # 4 security properties
        assert result['all_passed'] == True
        assert result['passed'] == 4
        assert result['failed'] == 0

    def test_mask_statistical_properties(self):
        """Test statistical properties of masks."""
        mask = generate_mask_from_seed(99999, (10000,), torch.float32)

        # Mean should be near 0
        mean = mask.mean().item()
        assert abs(mean) < 0.1

        # Std should be around 0.577 (uniform[-1,1])
        std = mask.std().item()
        expected_std = 1.0 / (3 ** 0.5)
        assert abs(std - expected_std) < 0.1

    def test_mask_deterministic(self):
        """Test that same seed produces same mask."""
        seed = 54321

        mask1 = generate_mask_from_seed(seed, (100,), torch.float32)
        mask2 = generate_mask_from_seed(seed, (100,), torch.float32)

        assert torch.allclose(mask1, mask2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
