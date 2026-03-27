"""
Label flipping attack implementation.
"""

import numpy as np
from typing import Optional
from torch.utils.data import DataLoader
import torch

from .base import BaseAttack


class LabelFlipAttack(BaseAttack):
    """
    Label flipping attack where adversary flips labels to targeted classes.

    This is a data poisoning attack where the attacker flips labels during
    local training to cause the global model to misclassify specific classes.
    """

    def __init__(self, config: dict):
        """
        Initialize label flip attack.

        Args:
            config: Attack configuration with keys:
                - flip_ratio: Fraction of labels to flip (default: 1.0)
                - source_class: Original class to flip from (default: 1 - fraud)
                - target_class: Target class to flip to (default: 0 - legitimate)
        """
        super().__init__(config)
        self.flip_ratio = config.get("flip_ratio", 1.0)
        self.source_class = config.get("source_class", 1)
        self.target_class = config.get("target_class", 0)

    def apply_attack(
        self,
        parameters: np.ndarray,
        local_data: Optional[DataLoader] = None,
        client_id: int = 0,
        global_model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """
        Apply label flip attack.

        This attack modifies the local training data labels before training,
        causing the model to learn incorrect mappings. After training with
        flipped labels, the resulting parameters are returned.

        Args:
            parameters: Original local model parameters
            local_data: Local training data (required for this attack)
            client_id: ID of the attacking client
            global_model: Global model reference

        Returns:
            Poisoned parameters after training with flipped labels
        """
        if local_data is None or global_model is None:
            # If no local data provided, return parameters modified directly
            # This simulates a stronger gradient manipulation attack
            return self._flip_gradients(parameters)

        # Get model from global_model reference
        model = global_model

        # Set model parameters
        model.set_parameters(torch.from_numpy(parameters).float())

        # Create poisoned dataset with flipped labels
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

    def _flip_gradients(self, parameters: np.ndarray) -> np.ndarray:
        """
        Flip gradients/parameters directly when data is not available.

        This is a simpler attack that manipulates parameters directly.

        Args:
            parameters: Original parameters

        Returns:
            Modified parameters
        """
        # For label flip, we negate gradients for classes involved in flipping
        # This is a simplified approach when data is unavailable
        poisoned = parameters.copy()

        # Apply perturbation
        perturbation = np.random.randn(*poisoned.shape) * 0.5
        poisoned += perturbation

        return poisoned

    def _create_poisoned_loader(self, original_loader: DataLoader) -> DataLoader:
        """
        Create a DataLoader with flipped labels.

        Args:
            original_loader: Original training data loader

        Returns:
            DataLoader with flipped labels
        """
        poisoned_x = []
        poisoned_y = []

        for x_batch, y_batch in original_loader:
            batch_size = len(y_batch)
            num_to_flip = int(batch_size * self.flip_ratio)

            # Get indices to flip
            flip_indices = np.where(y_batch.numpy() == self.source_class)[0]
            if len(flip_indices) > 0:
                # Select subset to flip
                num_flip = min(num_to_flip, len(flip_indices))
                selected_indices = np.random.choice(flip_indices, num_flip, replace=False)

                # Flip labels
                y_batch_flipped = y_batch.clone()
                y_batch_flipped[selected_indices] = self.target_class

                poisoned_x.append(x_batch)
                poisoned_y.append(y_batch_flipped)
            else:
                poisoned_x.append(x_batch)
                poisoned_y.append(y_batch)

        # Combine all batches
        all_x = torch.cat(poisoned_x, dim=0)
        all_y = torch.cat(poisoned_y, dim=0)

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(all_x, all_y),
            batch_size=original_loader.batch_size,
            shuffle=True,
        )
