"""
Inner Product Manipulation Attack

Maximizes negative inner product with honest updates.
Actively opposes the direction of honest client progress.

Strategy: argmin ⟨poisoned_update, honest_updates⟩
"""

from typing import Dict, List
import numpy as np
import torch

from .base_poison import ModelPoisoningAttack


class InnerProductAttack(ModelPoisoningAttack):
    """
    Inner product attack that actively opposes honest updates.

    This is an optimization-based attack that finds the worst-case
    perturbation by maximizing the negative inner product with honest
    client updates. More sophisticated than simple sign flipping.

    Algorithm:
    1. Start with honest update
    2. Iteratively perturb to minimize ⟨update, honest_updates⟩
    3. Constrain to realistic L2 norm to avoid detection

    Strengths:
    - Mathematically optimized for maximum disruption
    - Can be more powerful than sign flipping
    - Can constrain L2 norm to avoid detection

    Weaknesses:
    - Computationally expensive (requires optimization)
    - Requires access to honest updates (may not be realistic)
    - Still detectable via sophisticated defenses
    """

    def __init__(
        self,
        optimization_steps: int = 10,
        step_size: float = 0.1,
        l2_constraint: float = None
    ):
        """
        Initialize inner product attack.

        Args:
            optimization_steps: Number of optimization iterations
            step_size: Learning rate for gradient-based optimization
            l2_constraint: Maximum allowed L2 norm (None = no constraint)
        """
        super().__init__(attack_name="inner_product")
        self.optimization_steps = optimization_steps
        self.step_size = step_size
        self.l2_constraint = l2_constraint

    def poison_update(
        self,
        parameters: np.ndarray,
        layer_info: Dict[str, torch.Size],
        honest_updates: List[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Optimize update to maximize negative inner product with honest updates.

        Args:
            parameters: Honest gradient update
            layer_info: Layer shape information (not used directly)
            honest_updates: List of updates from honest clients

        Returns:
            Optimized poisoned parameters
        """
        self.attack_count += 1

        if honest_updates is None or len(honest_updates) == 0:
            # No honest updates available, fall back to sign flipping
            return -parameters

        # Compute average honest update direction
        avg_honest = np.mean(honest_updates, axis=0)

        # Initialize with honest update
        poisoned = parameters.copy().astype(np.float64)
        original_l2 = np.linalg.norm(poisoned)

        # Optimize to minimize inner product with honest direction
        for _ in range(self.optimization_steps):
            # Current inner product
            inner_prod = np.dot(poisoned, avg_honest)

            # Gradient of inner product w.r.t. poisoned is just avg_honest
            # To minimize inner product, move in negative direction
            gradient = avg_honest

            # Gradient descent on inner product
            poisoned = poisoned - self.step_size * gradient

            # Project to maintain L2 constraint
            if self.l2_constraint is not None:
                current_l2 = np.linalg.norm(poisoned)
                if current_l2 > self.l2_constraint:
                    poisoned = poisoned * (self.l2_constraint / current_l2)

        return poisoned.astype(np.float32)

    def __repr__(self) -> str:
        return f"InnerProductAttack(steps={self.optimization_steps}, lr={self.step_size})"
