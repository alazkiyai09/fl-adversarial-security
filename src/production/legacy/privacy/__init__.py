"""Privacy module for differential privacy and secure aggregation."""

from .differential_privacy import (
    DPSGDFactory,
    DPSGDOptimizer,
    PrivacyAccountant,
    clip_and_add_noise,
    compute_noise_multiplier,
    compute_sampling_probability,
)
from .secure_aggregation import (
    SecureAggregator,
    generate_random_mask,
    pairwise_mask,
    unmask_aggregate,
)

__all__ = [
    "DPSGDFactory",
    "DPSGDOptimizer",
    "PrivacyAccountant",
    "clip_and_add_noise",
    "compute_noise_multiplier",
    "compute_sampling_probability",
    "SecureAggregator",
    "generate_random_mask",
    "pairwise_mask",
    "unmask_aggregate",
]
