"""
Secure Aggregation for Federated Learning

Implementation of the Bonawitz et al. CCS 2017 secure aggregation protocol.

This package provides:
- Cryptographic primitives (DH key exchange, Shamir's secret sharing)
- Client and server protocol implementations
- Dropout recovery mechanisms
- Security verification tools
- Performance profiling utilities

Example usage:
    from src.secure_aggregation.legacy.simulation import run_simplified_simulation

    result = run_simplified_simulation(
        num_clients=10,
        model_size=1000,
        dropout_rate=0.2
    )

    print(f"Aggregate matches: {result['aggregate_matches']}")
"""

__version__ = "0.1.0"
__author__ = "Secure FL Researcher"

from .crypto import (
    generate_dh_keypair,
    compute_shared_secret,
    split_secret,
    reconstruct_secret,
    generate_mask_from_seed
)

from .protocol import (
    SecureAggregationClient,
    SecureAggregationServer,
    coordinate_recovery_protocol
)

from .aggregation import (
    apply_mask,
    cancel_mask,
    sum_updates
)

__all__ = [
    # Version
    '__version__',
    # Crypto
    'generate_dh_keypair',
    'compute_shared_secret',
    'split_secret',
    'reconstruct_secret',
    'generate_mask_from_seed',
    # Protocol
    'SecureAggregationClient',
    'SecureAggregationServer',
    'coordinate_recovery_protocol',
    # Aggregation
    'apply_mask',
    'cancel_mask',
    'sum_updates',
]
