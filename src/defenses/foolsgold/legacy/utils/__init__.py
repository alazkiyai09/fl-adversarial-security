"""
Utilities for FoolsGold experiments.
"""

from .similarity import (
    flatten_parameters,
    cosine_similarity,
    compute_pairwise_cosine_similarity,
    compute_adaptive_weights
)
from .metrics import (
    compute_accuracy,
    compute_attack_success_rate,
    track_client_contributions,
    DefenseMetrics
)

__all__ = [
    "flatten_parameters",
    "cosine_similarity",
    "compute_pairwise_cosine_similarity",
    "compute_adaptive_weights",
    "compute_accuracy",
    "compute_attack_success_rate",
    "track_client_contributions",
    "DefenseMetrics",
]
