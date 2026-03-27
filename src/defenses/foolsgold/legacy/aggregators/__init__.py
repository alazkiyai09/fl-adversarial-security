"""
FoolsGold Defense - Sybil-Resistant Federated Learning Aggregators
"""

from .base import BaseAggregator
from .foolsgold import FoolsGoldAggregator, compute_contribution_scores, compute_pairwise_cosine_similarity, foolsgold_aggregate
from .robust import KrumAggregator, MultiKrumAggregator, TrimmedMeanAggregator

__all__ = [
    "BaseAggregator",
    "FoolsGoldAggregator",
    "KrumAggregator",
    "MultiKrumAggregator",
    "TrimmedMeanAggregator",
    "compute_contribution_scores",
    "compute_pairwise_cosine_similarity",
    "foolsgold_aggregate",
]
