"""
Tests for Diffie-Hellman key agreement.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crypto.key_agreement import (
    generate_dh_keypair,
    compute_shared_secret,
    pairwise_key_agreement,
    verify_shared_secret
)


class TestKeyAgreement:
    """Test suite for Diffie-Hellman key agreement."""

    def test_generate_keypair(self):
        """Test key pair generation."""
        prime = 2**127 - 1
        generator = 2

        private_key, public_key = generate_dh_keypair(prime, generator)

        # Private key should be in valid range
        assert 1 <= private_key < prime - 1

        # Public key should be in valid range
        assert 0 <= public_key < prime

    def test_shared_secret_computation(self):
        """Test that both parties compute the same shared secret."""
        prime = 2**127 - 1
        generator = 2

        # Alice's key pair
        alice_private, alice_public = generate_dh_keypair(prime, generator)

        # Bob's key pair
        bob_private, bob_public = generate_dh_keypair(prime, generator)

        # Compute shared secrets
        alice_shared = compute_shared_secret(alice_private, bob_public, prime, generator)
        bob_shared = compute_shared_secret(bob_private, alice_public, prime, generator)

        # Should be equal
        assert alice_shared == bob_shared

    def test_pairwise_key_agreement(self):
        """Test pairwise key agreement with multiple clients."""
        prime = 2**127 - 1
        generator = 2
        client_ids = [0, 1, 2, 3, 4]

        # Client 0 establishes keys with all others
        shared_secrets = pairwise_key_agreement(0, client_ids, prime, generator)

        # Should have keys with all other clients
        assert len(shared_secrets) == len(client_ids) - 1
        assert 0 not in shared_secrets

        # All secrets should be non-zero
        for secret in shared_secrets.values():
            assert secret > 0

    def test_different_keys_different_sessions(self):
        """Test that different sessions produce different keys."""
        prime = 2**127 - 1
        generator = 2

        # Session 1
        priv1, pub1 = generate_dh_keypair(prime, generator)

        # Session 2
        priv2, pub2 = generate_dh_keypair(prime, generator)

        # Keys should be different (with very high probability)
        assert priv1 != priv2 or pub1 != pub2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
