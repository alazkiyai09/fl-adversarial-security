"""Reputation system modules for SignGuard."""

from src.defenses.signguard_full.legacy.reputation.base import ReputationSystem
from src.defenses.signguard_full.legacy.reputation.decay_reputation import DecayReputationSystem

__all__ = ["ReputationSystem", "DecayReputationSystem"]
