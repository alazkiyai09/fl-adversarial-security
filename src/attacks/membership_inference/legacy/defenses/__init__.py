"""Defense mechanisms against membership inference attacks."""

from .dp_defense import (
    DPTargetTrainer,
    test_dp_defense,
    analyze_privacy_utility_tradeoff,
    compute_effective_epsilon
)

__all__ = [
    'DPTargetTrainer',
    'test_dp_defense',
    'analyze_privacy_utility_tradeoff',
    'compute_effective_epsilon'
]
