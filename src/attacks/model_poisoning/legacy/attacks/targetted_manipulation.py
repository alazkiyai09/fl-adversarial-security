"""
Targeted Manipulation Attack

Modifies specific layer weights (e.g., last layer) to manipulate decisions.
Focuses attack on most critical layers for model predictions.

Strategy: poisoned_update[target_layer] = honest_update + perturbation
"""

from typing import Dict, List
import numpy as np
import torch

from .base_poison import ModelPoisoningAttack


class TargettedManipulationAttack(ModelPoisoningAttack):
    """
    Targeted manipulation attack that modifies specific layers.

    This attack focuses on manipulating specific layers rather than the
    entire model. The last layer is often targeted as it directly affects
    final predictions. More sophisticated than full-model attacks.

    Strengths:
    - More subtle than full-model attacks
    - Can target critical decision boundaries
    - Lower computational overhead

    Weaknesses:
    - Requires knowledge of layer structure
    - May be less effective than full-model poisoning
    - Target layers must be carefully chosen
    """

    def __init__(
        self,
        target_layers: List[str] = None,
        perturbation_scale: float = 5.0
    ):
        """
        Initialize targeted manipulation attack.

        Args:
            target_layers: List of layer names to target (e.g., ["fc2.weight"])
            perturbation_scale: Magnitude of perturbation to add
        """
        super().__init__(attack_name="targetted_manipulation")
        self.target_layers = target_layers or ["fc2.weight", "fc2.bias"]
        self.perturbation_scale = perturbation_scale

    def poison_update(
        self,
        parameters: np.ndarray,
        layer_info: Dict[str, torch.Size]
    ) -> np.ndarray:
        """
        Add perturbation to specific layers.

        Args:
            parameters: Flattened honest gradient update
            layer_info: Dictionary mapping layer names to (start_idx, end_idx, shape)

        Returns:
            Poisoned parameters with targeted modifications
        """
        self.attack_count += 1
        poisoned_params = parameters.copy()

        for layer_name in self.target_layers:
            if layer_name in layer_info:
                start_idx, end_idx, shape = layer_info[layer_name]
                layer_params = poisoned_params[start_idx:end_idx]

                # Add perturbation to this layer
                perturbation = np.random.randn(*layer_params.shape) * self.perturbation_scale
                poisoned_params[start_idx:end_idx] = layer_params + perturbation.flatten()

        return poisoned_params

    def __repr__(self) -> str:
        return f"TargettedManipulationAttack(layers={self.target_layers}, scale={self.perturbation_scale})"
