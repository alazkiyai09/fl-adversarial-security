"""Baseline defense implementations for SignGuard."""

from src.defenses.signguard_full.legacy.defenses.krum import KrumDefense
from src.defenses.signguard_full.legacy.defenses.trimmed_mean import TrimmedMeanDefense
from src.defenses.signguard_full.legacy.defenses.foolsgold import FoolsGoldDefense
from src.defenses.signguard_full.legacy.defenses.bulyan import BulyanDefense

__all__ = [
    "KrumDefense",
    "TrimmedMeanDefense",
    "FoolsGoldDefense",
    "BulyanDefense",
]
