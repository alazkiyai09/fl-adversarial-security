"""
Sign flipping attack implementation.
"""

import numpy as np
from typing import Optional
from torch.utils.data import DataLoader
import torch

from .base import BaseAttack


class SignFlipAttack(BaseAttack):
    """
    Sign flipping attack where adversary inverts their gradient updates.

    A Byzantine attack where the attacker negates their gradient (multiplies by -1)
    to maximize disruption to the aggregated model. This is particularly
    effective against mean-based aggregation like FedAvg.
    """

    def __init__(self, config: dict):
        """
        Initialize sign flip attack.

        Args:
            config: Attack configuration with keys:
                - scale: Additional scaling factor (default: 1.0)
                - smart_flip: Whether to use smart flipping (default: False)
        """
        super().__init__(config)
        self.scale = config.get("scale", 1.0)
        self.smart_flip = config.get("smart_flip", False)

    def apply_attack(
        self,
        parameters: np.ndarray,
        local_data: Optional[DataLoader] = None,
        client_id: int = 0,
        global_model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """
        Apply sign flipping attack.

        Args:
            parameters: Original local model parameters
            local_data: Local training data (not needed for this attack)
            client_id: ID of the attacking client
            global_model: Global model reference

        Returns:
            Sign-flipped parameters
        """
        if self.smart_flip and global_model is not None:
            # Smart flipping: identify important parameters and flip those
            poisoned = self._smart_sign_flip(parameters, global_model)
        else:
            # Naive flipping: flip all signs
            poisoned = -parameters * self.scale

        return poisoned.astype(np.float32)

    def _smart_sign_flip(
        self,
        parameters: np.ndarray,
        global_model: torch.nn.Module,
    ) -> np.ndarray:
        """
        Smart sign flipping that targets important parameters.

        Args:
            parameters: Original parameters
            global_model: Global model reference

        Returns:
            Smartly sign-flipped parameters
        """
        # Get parameter magnitudes to identify important ones
        magnitudes = np.abs(parameters)
        threshold = np.percentile(magnitudes, 50)  # Flip top 50%

        # Create mask for important parameters
        mask = magnitudes > threshold

        # Flip only important parameters
        poisoned = parameters.copy()
        poisoned[mask] = -poisoned[mask] * self.scale

        return poisoned


class AdaptiveSignFlipAttack(BaseAttack):
    """
    Adaptive sign flipping that adjusts based on other clients' updates.

    This attack tries to maximize disruption by adapting to the
    distribution of other clients' updates.
    """

    def __init__(self, config: dict):
        """
        Initialize adaptive sign flip attack.

        Args:
            config: Attack configuration with keys:
                - scale: Scaling factor (default: 1.0)
                - momentum: Momentum for adaptive direction (default: 0.9)
        """
        super().__init__(config)
        self.scale = config.get("scale", 1.0)
        self.momentum = config.get("momentum", 0.9)
        self._previous_update: Optional[np.ndarray] = None

    def apply_attack(
        self,
        parameters: np.ndarray,
        local_data: Optional[DataLoader] = None,
        client_id: int = 0,
        global_model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """
        Apply adaptive sign flipping attack.

        Args:
            parameters: Original local model parameters
            local_data: Local training data
            client_id: ID of the attacking client
            global_model: Global model reference

        Returns:
            Adaptively sign-flipped parameters
        """
        # Base sign flip
        poisoned = -parameters * self.scale

        # Apply momentum if available
        if self._previous_update is not None:
            poisoned = self.momentum * self._previous_update + (1 - self.momentum) * poisoned

        # Store for next round
        self._previous_update = poisoned

        return poisoned.astype(np.float32)

    def reset_state(self) -> None:
        """Reset attack state between experiments."""
        self._previous_update = None
