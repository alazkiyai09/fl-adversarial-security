"""
Gradient Scaling Attack

Amplifies model updates by a scaling factor λ.
Simple but effective: large λ can significantly slow convergence.

Strategy: poisoned_update = λ × honest_update
"""

from typing import Dict
import numpy as np
import torch

from .base_poison import ModelPoisoningAttack


class GradientScalingAttack(ModelPoisoningAttack):
    """
    Gradient scaling attack that amplifies updates by factor λ.

    This is one of the simplest model poisoning attacks. By sending
    amplified updates, malicious clients can dominate the aggregation
    and push the model in any direction they trained on.

    Strengths:
    - Simple to implement
    - Harder to detect than sign flipping (maintains direction)
    - Effective with large λ values

    Weaknesses:
    - More detectable at high λ values (L2 norm outlier)
    - Requires legitimate training to be useful
    """

    def __init__(self, scaling_factor: float = 10.0):
        """
        Initialize gradient scaling attack.

        Args:
            scaling_factor: λ multiplier for gradients (default: 10.0)
                           Higher = more powerful but more detectable
        """
        super().__init__(attack_name="gradient_scaling")
        self.scaling_factor = scaling_factor

    def poison_update(
        self,
        parameters: np.ndarray,
        layer_info: Dict[str, torch.Size]
    ) -> np.ndarray:
        """
        Scale gradients by factor λ.

        Args:
            parameters: Honest gradient update
            layer_info: Layer shape information (not used here)

        Returns:
            Poisoned (scaled) parameters
        """
        self.attack_count += 1
        poisoned_params = parameters * self.scaling_factor
        return poisoned_params

    def __repr__(self) -> str:
        return f"GradientScalingAttack(λ={self.scaling_factor})"
