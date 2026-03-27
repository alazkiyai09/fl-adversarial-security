"""
Gaussian Noise Attack

Adds random Gaussian noise to gradient updates.
Subtler attack: disrupts convergence without obvious patterns.

Strategy: poisoned_update = honest_update + N(0, σ²)
"""

from typing import Dict
import numpy as np
import torch

from .base_poison import ModelPoisoningAttack


class GaussianNoiseAttack(ModelPoisoningAttack):
    """
    Gaussian noise attack that adds random perturbation to updates.

    This attack adds zero-mean Gaussian noise to the gradients. Unlike
    sign flipping or gradient scaling, it doesn't have a clear direction,
    making it potentially harder to detect via simple similarity metrics.

    Strengths:
    - Less detectable than sign flipping (no clear pattern)
    - Can slow convergence effectively
    - Mimics natural gradient variance

    Weaknesses:
    - Less powerful than targeted attacks
    - May require higher noise levels to be effective
    - High σ increases detectability via L2 norm
    """

    def __init__(self, noise_std: float = 0.5):
        """
        Initialize Gaussian noise attack.

        Args:
            noise_std: Standard deviation σ of Gaussian noise (default: 0.5)
                      Higher = more disruptive but more detectable
        """
        super().__init__(attack_name="gaussian_noise")
        self.noise_std = noise_std

    def poison_update(
        self,
        parameters: np.ndarray,
        layer_info: Dict[str, torch.Size]
    ) -> np.ndarray:
        """
        Add Gaussian noise to gradients.

        Args:
            parameters: Honest gradient update
            layer_info: Layer shape information (not used here)

        Returns:
            Poisoned (noisy) parameters
        """
        self.attack_count += 1
        noise = np.random.normal(0, self.noise_std, parameters.shape)
        poisoned_params = parameters + noise
        return poisoned_params

    def __repr__(self) -> str:
        return f"GaussianNoiseAttack(σ={self.noise_std})"
