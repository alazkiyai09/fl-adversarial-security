"""
Attack implementations for testing FoolsGold defense.
"""

from .sybil import SybilAttack, generate_sybil_updates
from .collusion import CollusionAttack
from .label_flipping import LabelFlippingAttack

__all__ = [
    "SybilAttack",
    "CollusionAttack",
    "LabelFlippingAttack",
    "generate_sybil_updates",
]
