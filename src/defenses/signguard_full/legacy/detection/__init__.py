"""Anomaly detection modules for SignGuard."""

from src.defenses.signguard_full.legacy.detection.base import AnomalyDetector
from src.defenses.signguard_full.legacy.detection.magnitude_detector import L2NormDetector
from src.defenses.signguard_full.legacy.detection.direction_detector import CosineSimilarityDetector
from src.defenses.signguard_full.legacy.detection.score_detector import LossDeviationDetector
from src.defenses.signguard_full.legacy.detection.ensemble import EnsembleDetector

__all__ = [
    "AnomalyDetector",
    "EnsembleDetector",
    "L2NormDetector",
    "CosineSimilarityDetector",
    "LossDeviationDetector",
]
