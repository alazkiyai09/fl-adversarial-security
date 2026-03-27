"""Attack implementations for SignGuard."""

from src.defenses.signguard_full.legacy.attacks.base import Attack
from src.defenses.signguard_full.legacy.attacks.label_flip import LabelFlipAttack
from src.defenses.signguard_full.legacy.attacks.backdoor import BackdoorAttack
from src.defenses.signguard_full.legacy.attacks.model_poison import ModelPoisonAttack

__all__ = [
    "Attack",
    "LabelFlipAttack",
    "BackdoorAttack",
    "ModelPoisonAttack",
]
