"""
Aggregation operations for secure aggregation.
"""

from .masked_update import (
    apply_mask,
    cancel_mask,
    verify_mask_cancellation
)

from .aggregator import (
    sum_updates,
    compute_average,
    SecureAggregator
)

__all__ = [
    'apply_mask',
    'cancel_mask',
    'verify_mask_cancellation',
    'sum_updates',
    'compute_average',
    'SecureAggregator',
]
