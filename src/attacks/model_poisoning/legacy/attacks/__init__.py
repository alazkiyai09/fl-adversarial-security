"""
Model Poisoning Attacks for Federated Learning

This module implements various model poisoning strategies that directly manipulate
gradients/weights during federated aggregation (not training data).

Reference: Bhagoji et al., "Analyzing Federated Learning through an Adversarial Lens" (ICML 2019)
"""

from .base_poison import ModelPoisoningAttack
from .gradient_scaling import GradientScalingAttack
from .sign_flipping import SignFlippingAttack
from .gaussian_noise import GaussianNoiseAttack
from .targetted_manipulation import TargettedManipulationAttack
from .inner_product import InnerProductAttack

__all__ = [
    "ModelPoisoningAttack",
    "GradientScalingAttack",
    "SignFlippingAttack",
    "GaussianNoiseAttack",
    "TargettedManipulationAttack",
    "InnerProductAttack",
]
