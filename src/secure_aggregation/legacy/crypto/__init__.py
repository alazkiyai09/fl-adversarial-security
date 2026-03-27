"""
Cryptographic primitives for secure aggregation.

This module provides the foundational cryptographic building blocks:
- Diffie-Hellman key agreement
- Shamir's secret sharing
- Pseudo-random functions for mask generation
"""

from .key_agreement import (
    generate_dh_keypair,
    compute_shared_secret,
    pairwise_key_agreement
)

from .secret_sharing import (
    split_secret,
    reconstruct_secret,
    verify_reconstruction
)

from .prf import (
    prf,
    generate_mask_from_seed
)

__all__ = [
    # Key agreement
    'generate_dh_keypair',
    'compute_shared_secret',
    'pairwise_key_agreement',
    # Secret sharing
    'split_secret',
    'reconstruct_secret',
    'verify_reconstruction',
    # PRF
    'prf',
    'generate_mask_from_seed',
]
