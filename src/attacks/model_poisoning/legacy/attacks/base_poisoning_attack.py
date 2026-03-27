"""
Base class for model poisoning attacks.

This is a duplicate import to ensure compatibility.
All poisoning strategies inherit from this abstract base class.
"""

from .base_poison import ModelPoisoningAttack

__all__ = ["ModelPoisoningAttack"]
