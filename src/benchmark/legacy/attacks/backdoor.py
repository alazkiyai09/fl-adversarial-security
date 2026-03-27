"""
Backdoor attack implementation.
"""

import numpy as np
from typing import Optional
from torch.utils.data import DataLoader
import torch

from .base import BaseAttack


class BackdoorAttack(BaseAttack):
    """
    Backdoor attack where adversary implants a hidden trigger in the model.

    The attacker adds a pattern (trigger) to inputs and trains the model to
    misclassify to a target class when the trigger is present.
    """

    def __init__(self, config: dict):
        """
        Initialize backdoor attack.

        Args:
            config: Attack configuration with keys:
                - target_class: Target class for backdoor (default: 0)
                - poison_ratio: Fraction of data to poison (default: 0.5)
                - trigger_pattern: Type of trigger ('pixel', 'pattern', 'gaussian')
                - trigger_scale: Scale of trigger perturbation (default: 2.0)
        """
        super().__init__(config)
        self.target_class = config.get("target_class", 0)
        self.poison_ratio = config.get("poison_ratio", 0.5)
        self.trigger_pattern = config.get("trigger_pattern", "gaussian")
        self.trigger_scale = config.get("trigger_scale", 2.0)

    def apply_attack(
        self,
        parameters: np.ndarray,
        local_data: Optional[DataLoader] = None,
        client_id: int = 0,
        global_model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """
        Apply backdoor attack.

        Args:
            parameters: Original local model parameters
            local_data: Local training data (required for this attack)
            client_id: ID of the attacking client
            global_model: Global model reference

        Returns:
            Poisoned parameters with backdoor implanted
        """
        if local_data is None or global_model is None:
            # Direct parameter manipulation when data unavailable
            return self._implant_backdoor_direct(parameters)

        # Get model from global_model reference
        model = global_model

        # Set model parameters
        model.set_parameters(torch.from_numpy(parameters).float())

        # Create poisoned dataset with backdoor trigger
        poisoned_data = self._create_poisoned_loader(local_data)

        # Train with poisoned data
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(5):  # Local epochs
            for x_batch, y_batch in poisoned_data:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # Return poisoned parameters
        return model.get_parameters().detach().cpu().numpy()

    def _implant_backdoor_direct(self, parameters: np.ndarray) -> np.ndarray:
        """
        Implant backdoor directly into parameters when data is unavailable.

        Args:
            parameters: Original parameters

        Returns:
            Backdoored parameters
        """
        # Add perturbation that creates backdoor behavior
        # This is a simplified approach
        poisoned = parameters.copy()

        # Apply targeted perturbation
        if self.trigger_pattern == "gaussian":
            perturbation = np.random.randn(*poisoned.shape) * self.trigger_scale * 0.1
        else:
            perturbation = np.ones_like(poisoned) * self.trigger_scale * 0.1

        poisoned += perturbation

        return poisoned

    def _create_poisoned_loader(self, original_loader: DataLoader) -> DataLoader:
        """
        Create a DataLoader with backdoor trigger.

        Args:
            original_loader: Original training data loader

        Returns:
            DataLoader with backdoor trigger
        """
        poisoned_x = []
        poisoned_y = []

        for x_batch, y_batch in original_loader:
            batch_size = len(y_batch)
            num_to_poison = int(batch_size * self.poison_ratio)

            # Create copies
            x_poisoned = x_batch.clone()
            y_poisoned = y_batch.clone()

            # Select samples to poison
            poison_indices = np.random.choice(batch_size, num_to_poison, replace=False)

            # Add trigger to poisoned samples
            for idx in poison_indices:
                x_poisoned[idx] = self._add_trigger(x_poisoned[idx])
                y_poisoned[idx] = self.target_class

            poisoned_x.append(x_poisoned)
            poisoned_y.append(y_poisoned)

        # Combine all batches
        all_x = torch.cat(poisoned_x, dim=0)
        all_y = torch.cat(poisoned_y, dim=0)

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(all_x, all_y),
            batch_size=original_loader.batch_size,
            shuffle=True,
        )

    def _add_trigger(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add backdoor trigger to input.

        Args:
            x: Input tensor

        Returns:
            Input with trigger added
        """
        if self.trigger_pattern == "gaussian":
            # Add Gaussian noise trigger
            trigger = torch.randn_like(x) * self.trigger_scale
            return x + trigger
        elif self.trigger_pattern == "pattern":
            # Add pattern trigger (e.g., specific features)
            trigger = torch.ones_like(x) * self.trigger_scale
            # Apply to subset of features
            trigger[:, :len(trigger)//2] = 0
            return x + trigger
        else:
            # Default: simple additive trigger
            return x + self.trigger_scale
