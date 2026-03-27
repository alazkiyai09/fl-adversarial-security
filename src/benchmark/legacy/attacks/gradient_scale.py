"""
Gradient scaling attack implementation.
"""

import numpy as np
from typing import Optional
from torch.utils.data import DataLoader
import torch

from .base import BaseAttack


class GradientScaleAttack(BaseAttack):
    """
    Gradient scaling attack where adversary scales their gradient updates.

    A Byzantine attack where the attacker multiplies their gradient by a
    large scaling factor to dominate the aggregation and push the model
    in a malicious direction.
    """

    def __init__(self, config: dict):
        """
        Initialize gradient scaling attack.

        Args:
            config: Attack configuration with keys:
                - scale_factor: Scaling factor for gradients (default: 10.0)
        """
        super().__init__(config)
        self.scale_factor = config.get("scale_factor", 10.0)

    def apply_attack(
        self,
        parameters: np.ndarray,
        local_data: Optional[DataLoader] = None,
        client_id: int = 0,
        global_model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """
        Apply gradient scaling attack.

        Args:
            parameters: Original local model parameters
            local_data: Local training data (not needed for this attack)
            client_id: ID of the attacking client
            global_model: Global model reference

        Returns:
            Scaled parameters
        """
        # Scale parameters directly
        poisoned = parameters * self.scale_factor

        return poisoned.astype(np.float32)


class DirectedGradientScaleAttack(BaseAttack):
    """
    Directed gradient scaling attack that scales gradients in a specific direction.

    Unlike simple scaling, this attack scales gradients along a specific
    adversarial direction to maximize damage.
    """

    def __init__(self, config: dict):
        """
        Initialize directed gradient scaling attack.

        Args:
            config: Attack configuration with keys:
                - scale_factor: Scaling factor for gradients (default: 10.0)
                - direction_type: Type of direction ('random', 'opposite', 'target')
                - target_class: Target class for directed attack (default: 0)
        """
        super().__init__(config)
        self.scale_factor = config.get("scale_factor", 10.0)
        self.direction_type = config.get("direction_type", "opposite")
        self.target_class = config.get("target_class", 0)
        self._direction: Optional[np.ndarray] = None

    def apply_attack(
        self,
        parameters: np.ndarray,
        local_data: Optional[DataLoader] = None,
        client_id: int = 0,
        global_model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """
        Apply directed gradient scaling attack.

        Args:
            parameters: Original local model parameters
            local_data: Local training data
            client_id: ID of the attacking client
            global_model: Global model reference

        Returns:
            Scaled parameters in adversarial direction
        """
        # Determine attack direction
        if self._direction is None or self.direction_type == "random":
            self._direction = self._generate_direction(parameters.shape)

        elif self.direction_type == "opposite":
            # Opposite direction of current parameters
            norm = np.linalg.norm(parameters) + 1e-8
            self._direction = -parameters / norm

        elif self.direction_type == "target":
            # Direction towards misclassifying target class
            self._direction = self._generate_target_direction(parameters)

        # Scale parameters in attack direction
        direction_norm = np.linalg.norm(self._direction) + 1e-8
        poisoned = parameters + (self._direction / direction_norm) * self.scale_factor * np.linalg.norm(parameters)

        return poisoned.astype(np.float32)

    def _generate_direction(self, shape: tuple) -> np.ndarray:
        """
        Generate random attack direction.

        Args:
            shape: Shape of direction vector

        Returns:
            Random unit direction vector
        """
        direction = np.random.randn(*shape)
        norm = np.linalg.norm(direction) + 1e-8
        return direction / norm

    def _generate_target_direction(self, parameters: np.ndarray) -> np.ndarray:
        """
        Generate direction towards target class misclassification.

        Args:
            parameters: Current parameters

        Returns:
            Targeted direction vector
        """
        # Create direction that emphasizes features for target class
        # This is a simplified approach
        direction = np.random.randn(*parameters.shape)
        direction = direction * np.sign(parameters)  # Same sign, scaled
        return direction
