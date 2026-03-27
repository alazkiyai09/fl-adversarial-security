"""
Diffie-Hellman key agreement for secure aggregation.

Implements pairwise Diffie-Hellman key exchange between clients
to establish shared secrets for seed encryption.
"""

from typing import Dict, List, Tuple
import secrets


def generate_dh_keypair(prime: int, generator: int) -> Tuple[int, int]:
    """
    Generate a Diffie-Hellman key pair.

    Args:
        prime: The prime modulus for the DH group
        generator: The generator for the DH group

    Returns:
        Tuple of (private_key, public_key)
        - private_key: Random value in [1, prime-2]
        - public_key: generator^private_key mod prime
    """
    # Private key: cryptographically secure random value in [1, prime-2]
    private_key = secrets.randbelow(prime - 2) + 1

    # Public key: g^private mod p
    public_key = pow(generator, private_key, prime)

    return private_key, public_key


def compute_shared_secret(
    private_key: int,
    peer_public_key: int,
    prime: int,
    generator: int
) -> int:
    """
    Compute the shared secret using peer's public key.

    Args:
        private_key: This client's private key
        peer_public_key: Peer's public key
        prime: The prime modulus for the DH group
        generator: The generator for the DH group (unused in computation, kept for interface consistency)

    Returns:
        shared_secret: peer_public_key^private_key mod prime
    """
    shared_secret = pow(peer_public_key, private_key, prime)
    return shared_secret


def pairwise_key_agreement(
    client_id: int,
    all_client_ids: List[int],
    prime: int,
    generator: int
) -> Dict[int, int]:
    """
    Simulate pairwise key agreement with all other clients.

    In a real implementation, this would involve actual communication.
    Here we simulate by generating all key pairs and computing shared secrets.

    Args:
        client_id: This client's ID
        all_client_ids: List of all client IDs in the system
        prime: The prime modulus for the DH group
        generator: The generator for the DH group

    Returns:
        Dictionary mapping peer_client_id -> shared_secret
    """
    shared_secrets = {}

    # Generate this client's key pair
    my_private, my_public = generate_dh_keypair(prime, generator)

    # In a real system, we would exchange public keys with all peers
    # For simulation, we'll generate all key pairs and compute shared secrets
    all_keypairs = {}
    for pid in all_client_ids:
        if pid != client_id:
            priv, pub = generate_dh_keypair(prime, generator)
            all_keypairs[pid] = (priv, pub)

    # Compute shared secret with each peer
    for peer_id in all_client_ids:
        if peer_id != client_id:
            peer_private, peer_public = all_keypairs[peer_id]
            # Shared secret = peer_public^my_private mod p
            shared_secret = compute_shared_secret(my_private, peer_public, prime, generator)
            shared_secrets[peer_id] = shared_secret

    return shared_secrets


def verify_shared_secret(
    secret1: int,
    secret2: int
) -> bool:
    """
    Verify that two computed shared secrets match.

    Args:
        secret1: First shared secret
        secret2: Second shared secret

    Returns:
        True if secrets match, False otherwise
    """
    return secret1 == secret2
