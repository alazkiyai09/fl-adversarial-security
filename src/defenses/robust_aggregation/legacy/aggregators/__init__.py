"""
Byzantine-Robust Aggregators for Federated Learning

This module implements aggregation methods resilient to Byzantine (malicious) clients.
"""

from .base import RobustAggregator
from .median import CoordinateWiseMedian
from .trimmed_mean import TrimmedMean
from .krum import Krum, MultiKrum
from .bulyan import Bulyan

__all__ = [
    'RobustAggregator',
    'CoordinateWiseMedian',
    'TrimmedMean',
    'Krum',
    'MultiKrum',
    'Bulyan',
]
