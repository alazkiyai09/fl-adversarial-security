"""
Server-side components for federated learning with attack monitoring.
"""

from .aggregation import FedAvgWithAttackTracking
from .detection import AttackDetector

__all__ = ["FedAvgWithAttackTracking", "AttackDetector"]
