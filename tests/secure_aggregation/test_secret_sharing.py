"""
Tests for Shamir's secret sharing.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crypto.secret_sharing import (
    split_secret,
    reconstruct_secret,
    verify_reconstruction,
    verify_threshold_property
)


class TestSecretSharing:
    """Test suite for Shamir's secret sharing."""

    def test_split_and_reconstruct(self):
        """Test basic split and reconstruction."""
        secret = 12345
        threshold = 3
        num_shares = 5
        prime = 2**127 - 1

        # Split secret
        shares = split_secret(secret, threshold, num_shares, prime)

        # Should have correct number of shares
        assert len(shares) == num_shares

        # Reconstruct with threshold shares
        reconstructed = reconstruct_secret(shares[:threshold], prime)

        # Should match original
        assert reconstructed == secret

    def test_reconstruct_with_different_subsets(self):
        """Test reconstruction with different share subsets."""
        secret = 54321
        threshold = 3
        num_shares = 6
        prime = 2**127 - 1

        shares = split_secret(secret, threshold, num_shares, prime)

        # Try different combinations
        import itertools

        for subset in itertools.combinations(shares, threshold):
            reconstructed = reconstruct_secret(list(subset), prime)
            assert reconstructed == secret

    def test_insufficient_shares_fail(self):
        """Test that insufficient shares cannot reconstruct."""
        secret = 99999
        threshold = 5
        num_shares = 8
        prime = 2**127 - 1

        shares = split_secret(secret, threshold, num_shares, prime)

        # Try with threshold - 1 shares (should fail to get correct secret)
        # Note: It might reconstruct some value, but not the original
        insufficient_shares = shares[:threshold - 1]
        reconstructed = reconstruct_secret(insufficient_shares, prime)

        # Very likely to be wrong
        assert reconstructed != secret

    def test_verify_reconstruction(self):
        """Test the verification function."""
        secret = 11111
        threshold = 3
        num_shares = 5
        prime = 2**127 - 1

        shares = split_secret(secret, threshold, num_shares, prime)

        # Should pass with threshold shares
        assert verify_reconstruction(shares[:threshold], secret, prime)

        # Should fail with insufficient shares
        assert not verify_reconstruction(shares[:threshold - 1], secret, prime)

    def test_threshold_property(self):
        """Test that threshold property holds."""
        secret = 77777
        threshold = 4
        num_shares = 7
        prime = 2**127 - 1

        # Should verify that t-1 shares don't leak the secret
        assert verify_threshold_property(secret, threshold, num_shares, prime)

    def test_zero_secret(self):
        """Test sharing and reconstructing zero."""
        secret = 0
        threshold = 2
        num_shares = 4
        prime = 2**127 - 1

        shares = split_secret(secret, threshold, num_shares, prime)
        reconstructed = reconstruct_secret(shares[:threshold], prime)

        assert reconstructed == secret

    def test_large_secret(self):
        """Test with a large secret value."""
        secret = 2**100
        threshold = 3
        num_shares = 5
        prime = 2**127 - 1

        shares = split_secret(secret, threshold, num_shares, prime)
        reconstructed = reconstruct_secret(shares[:threshold], prime)

        assert reconstructed == secret


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
