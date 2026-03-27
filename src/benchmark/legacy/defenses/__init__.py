"""
Defense implementations for robust federated learning aggregation.
"""

from .base import BaseDefense, DefenseResult
from .robust_aggregation import (
    FedAvgDefense,
    MedianDefense,
    TrimmedMeanDefense,
    KrumDefense,
    MultiKrumDefense,
    BulyanDefense,
    create_defense,
)
from .foolsgold import FoolsGoldDefense, FoolsGoldSimpleDefense
from .anomaly_detection import (
    AnomalyDetectionDefense,
    ClusteringDefense,
)

__all__ = [
    "BaseDefense",
    "DefenseResult",
    "FedAvgDefense",
    "MedianDefense",
    "TrimmedMeanDefense",
    "KrumDefense",
    "MultiKrumDefense",
    "BulyanDefense",
    "FoolsGoldDefense",
    "FoolsGoldSimpleDefense",
    "AnomalyDetectionDefense",
    "ClusteringDefense",
    "create_defense",
]
