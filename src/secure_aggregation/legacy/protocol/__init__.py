"""
Protocol implementation for secure aggregation.
"""

from .client import SecureAggregationClient
from .server import SecureAggregationServer
from .dropout_recovery import (
    coordinate_recovery_protocol,
    validate_threshold_sufficient
)

__all__ = [
    'SecureAggregationClient',
    'SecureAggregationServer',
    'coordinate_recovery_protocol',
    'validate_threshold_sufficient',
]
