"""
Sign Flipping Attack

Reverses the direction of gradient updates.
One of the most powerful attacks: directly opposes honest updates.

Strategy: poisoned_update = -1 × honest_update
"""

from typing import Dict
import numpy as np
import torch

from .base_poison import ModelPoisoningAttack


class SignFlippingAttack(ModelPoisoningAttack):
    """
    Sign flipping attack that reverses gradient direction.

    This is a highly effective model poisoning attack. By flipping the
    sign of updates, the malicious client directly opposes the progress
    made by honest clients, effectively canceling out their contributions.

    Strengths:
    - Extremely disruptive to convergence
    - Simple to implement
    - Can prevent convergence entirely with sufficient attackers

    Weaknesses:
    - Highly detectable via cosine similarity (~-1 correlation)
    - L2 norm may look normal, making it harder to detect by magnitude alone
    - Obvious in aggregated model performance degradation
    """

    def __init__(self, factor: float = -1.0):
        """
        Initialize sign flipping attack.

        Args:
            factor: Multiplier for gradients (default: -1.0 for sign flip)
                    Could use other negative values for partial flipping
        """
        super().__init__(attack_name="sign_flipping")
        self.factor = factor

    def poison_update(
        self,
        parameters: np.ndarray,
        layer_info: Dict[str, torch.Size]
    ) -> np.ndarray:
        """
        Flip the sign of gradient updates.

        Args:
            parameters: Honest gradient update
            layer_info: Layer shape information (not used here)

        Returns:
            Poisoned (sign-flipped) parameters
        """
        self.attack_count += 1
        poisoned_params = parameters * self.factor
        return poisoned_params

    def __repr__(self) -> str:
        return f"SignFlippingAttack(factor={self.factor})"
