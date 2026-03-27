"""
Gaussian noise attack implementation.
"""

import numpy as np
from typing import Optional
from torch.utils.data import DataLoader
import torch

from .base import BaseAttack


class GaussianNoiseAttack(BaseAttack):
    """
    Gaussian noise attack where adversary adds random noise to their updates.

    A Byzantine attack where the attacker adds Gaussian noise to their
    gradient updates, making the aggregation less effective.
    """

    def __init__(self, config: dict):
        """
        Initialize Gaussian noise attack.

        Args:
            config: Attack configuration with keys:
                - mean: Mean of noise distribution (default: 0.0)
                - std: Standard deviation of noise (default: 1.0)
                - relative: Whether std is relative to parameter norm (default: True)
                - clip: Whether to clip values (default: False)
        """
        super().__init__(config)
        self.mean = config.get("mean", 0.0)
        self.std = config.get("std", 1.0)
        self.relative = config.get("relative", True)
        self.clip = config.get("clip", False)

    def apply_attack(
        self,
        parameters: np.ndarray,
        local_data: Optional[DataLoader] = None,
        client_id: int = 0,
        global_model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """
        Apply Gaussian noise attack.

        Args:
            parameters: Original local model parameters
            local_data: Local training data (not needed for this attack)
            client_id: ID of the attacking client
            global_model: Global model reference

        Returns:
            Parameters with added Gaussian noise
        """
        # Determine noise standard deviation
        if self.relative:
            param_std = np.std(parameters) + 1e-8
            actual_std = self.std * param_std
        else:
            actual_std = self.std

        # Generate noise
        noise = np.random.normal(
            loc=self.mean,
            scale=actual_std,
            size=parameters.shape
        )

        # Add noise to parameters
        poisoned = parameters + noise

        # Optional clipping
        if self.clip:
            # Clip to reasonable range
            poisoned = np.clip(poisoned, -10 * actual_std, 10 * actual_std)

        return poisoned.astype(np.float32)


class TargetedGaussianNoiseAttack(BaseAttack):
    """
    Targeted Gaussian noise that focuses on specific parameters.

    Instead of adding noise to all parameters, this attack targets
    the most important parameters to maximize impact.
    """

    def __init__(self, config: dict):
        """
        Initialize targeted Gaussian noise attack.

        Args:
            config: Attack configuration with keys:
                - std: Standard deviation of noise (default: 1.0)
                - target_ratio: Fraction of parameters to target (default: 0.5)
                - selection_method: How to select targets ('magnitude', 'random')
        """
        super().__init__(config)
        self.std = config.get("std", 1.0)
        self.target_ratio = config.get("target_ratio", 0.5)
        self.selection_method = config.get("selection_method", "magnitude")

    def apply_attack(
        self,
        parameters: np.ndarray,
        local_data: Optional[DataLoader] = None,
        client_id: int = 0,
        global_model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """
        Apply targeted Gaussian noise attack.

        Args:
            parameters: Original local model parameters
            local_data: Local training data
            client_id: ID of the attacking client
            global_model: Global model reference

        Returns:
            Parameters with targeted noise added
        """
        poisoned = parameters.copy()

        # Select target parameters
        if self.selection_method == "magnitude":
            # Target parameters with largest magnitude
            magnitudes = np.abs(parameters)
            threshold = np.percentile(magnitudes, 100 * (1 - self.target_ratio))
            mask = magnitudes > threshold
        else:
            # Random selection
            num_params = parameters.size
            num_targets = int(num_params * self.target_ratio)
            flat_mask = np.zeros(num_params, dtype=bool)
            target_indices = np.random.choice(num_params, num_targets, replace=False)
            flat_mask[target_indices] = True
            mask = flat_mask.reshape(parameters.shape)

        # Add noise to targeted parameters
        param_std = np.std(parameters[mask]) + 1e-8
        noise = np.random.normal(
            loc=0.0,
            scale=self.std * param_std,
            size=parameters.shape
        )

        poisoned[mask] += noise[mask]

        return poisoned.astype(np.float32)
