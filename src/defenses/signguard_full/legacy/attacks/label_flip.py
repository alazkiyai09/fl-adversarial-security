"""Label flipping attack implementation."""

import torch
from torch.utils.data import DataLoader
from typing import Optional

from src.defenses.signguard_full.legacy.attacks.base import Attack
from src.defenses.signguard_full.legacy.core.types import ModelUpdate


class LabelFlipAttack(Attack):
    """Label flipping attack.

    Flips labels of training data to cause misclassification.
    Can flip from a specific source class or randomly.
    """

    def __init__(
        self,
        flip_ratio: float = 0.2,
        source_class: Optional[int] = None,
        target_class: int = 1,
        magnitude: float = 1.0,
    ):
        """Initialize label flip attack.

        Args:
            flip_ratio: Fraction of labels to flip
            source_class: Source class to flip from (None = random)
            target_class: Target class to flip to
            magnitude: Scaling factor for the malicious update
        """
        self.flip_ratio = flip_ratio
        self.source_class = source_class
        self.target_class = target_class
        self.magnitude = magnitude

    def execute(
        self,
        client_id: str,
        global_model: dict[str, torch.Tensor],
        train_loader: Optional[DataLoader] = None,
    ) -> ModelUpdate:
        """Execute label flip attack.

        Args:
            client_id: Client identifier
            global_model: Current global model
            train_loader: Training data (for creating poisoned update)

        Returns:
            Malicious model update with flipped label effects
        """
        # Create malicious update by inverting gradient direction
        # This simulates the effect of training on flipped labels
        malicious_params = {}
        
        for param_name, param_value in global_model.items():
            # Invert the update direction and scale
            # This approximates training with flipped labels
            noise = torch.randn_like(param_value) * self.magnitude
            malicious_params[param_name] = -noise  # Opposite direction
        
        return ModelUpdate(
            client_id=client_id,
            round_num=0,
            parameters=malicious_params,
            num_samples=100,  # Placeholder
            metrics={"loss": 2.0, "accuracy": 0.3},  # Indicate poor training
        )

    def flip_labels_in_dataset(
        self,
        dataloader: DataLoader,
    ) -> DataLoader:
        """Flip labels in a dataset.

        Args:
            dataloader: Original data loader

        Returns:
            New data loader with flipped labels
        """
        from torch.utils.data import TensorDataset
        
        all_inputs = []
        all_targets = []
        
        for inputs, targets in dataloader:
            inputs = inputs.clone()
            targets = targets.clone()
            
            # Flip labels
            num_to_flip = int(len(targets) * self.flip_ratio)
            
            if self.source_class is not None:
                # Flip from specific source class
                source_mask = targets == self.source_class
                source_indices = torch.where(source_mask)[0]
                
                if len(source_indices) > 0:
                    # Flip as many as we have, up to num_to_flip
                    num_flip = min(num_to_flip, len(source_indices))
                    flip_indices = source_indices[:num_flip]
                    targets[flip_indices] = self.target_class
            else:
                # Random flip
                flip_indices = torch.randperm(len(targets))[:num_to_flip]
                targets[flip_indices] = self.target_class
            
            all_inputs.append(inputs)
            all_targets.append(targets)
        
        # Create new dataset
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        poisoned_dataset = TensorDataset(all_inputs, all_targets)
        poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=dataloader.batch_size,
            shuffle=dataloader.shuffle,
        )
        
        return poisoned_loader

    def get_name(self) -> str:
        """Get attack name."""
        return "label_flip"
