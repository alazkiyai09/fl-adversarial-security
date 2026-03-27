"""
Label flipping attacks package.

This package provides implementations of various label flipping attacks
for federated learning systems.
"""

from .label_flip import (
    random_flip,
    targeted_flip,
    inverse_flip,
    apply_attack,
    LabelFlipAttack,
    create_attack,
)

__all__ = [
    "random_flip",
    "targeted_flip",
    "inverse_flip",
    "apply_attack",
    "LabelFlipAttack",
    "create_attack",
]
