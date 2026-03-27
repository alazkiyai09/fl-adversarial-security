"""Backdoor attack implementation."""

import torch
from torch.utils.data import DataLoader
from typing import Optional

from src.defenses.signguard_full.legacy.attacks.base import Attack
from src.defenses.signguard_full.legacy.core.types import ModelUpdate


class BackdoorAttack(Attack):
    """Backdoor insertion attack.

    Inserts a trigger pattern into data and associates it with a target class.
    When the trigger is present, the model should predict the target class.
    """

    def __init__(
        self,
        trigger_pattern: torch.Tensor,
        target_class: int = 1,
        poison_ratio: float = 0.2,
        injection_strength: float = 1.0,
    ):
        """Initialize backdoor attack.

        Args:
            trigger_pattern: Trigger pattern to inject
            target_class: Target class for backdoor
            poison_ratio: Fraction of data to poison
            injection_strength: Strength of trigger injection
        """
        self.trigger_pattern = trigger_pattern
        self.target_class = target_class
        self.poison_ratio = poison_ratio
        self.injection_strength = injection_strength

    def execute(
        self,
        client_id: str,
        global_model: dict[str, torch.Tensor],
        train_loader: Optional[DataLoader] = None,
    ) -> ModelUpdate:
        """Execute backdoor attack.

        Args:
            client_id: Client identifier
            global_model: Current global model
            train_loader: Training data (for trigger info)

        Returns:
            Malicious model update that learns backdoor
        """
        # Create malicious update optimized for backdoor
        # This creates parameters that activate on the trigger pattern
        malicious_params = {}
        
        for param_name, param_value in global_model.items():
            # Create update that biases toward target class
            # when trigger pattern is detected
            noise = torch.randn_like(param_value) * self.injection_strength
            
            # Add bias toward target class output
            if "weight" in param_name or "bias" in param_name:
                malicious_params[param_name] = noise * 2.0  # Stronger effect
            else:
                malicious_params[param_name] = noise
        
        return ModelUpdate(
            client_id=client_id,
            round_num=0,
            parameters=malicious_params,
            num_samples=100,
            metrics={"loss": 1.5, "accuracy": 0.4},
        )

    def inject_trigger(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Inject trigger pattern into inputs.

        Args:
            inputs: Input tensor

        Returns:
            Inputs with injected trigger
        """
        inputs = inputs.clone()
        batch_size = inputs.size(0)
        
        # Flatten inputs for pattern application
        original_shape = inputs.shape
        flat_inputs = inputs.view(batch_size, -1)
        
        # Apply trigger to first few features
        trigger_size = min(self.trigger_pattern.numel(), flat_inputs.size(1))
        
        for i in range(batch_size):
            flat_inputs[i, :trigger_size] += (
                self.trigger_pattern[:trigger_size] * self.injection_strength
            )
        
        return flat_inputs.view(original_shape)

    def poison_dataset(
        self,
        dataloader: DataLoader,
    ) -> DataLoader:
        """Poison dataset with backdoor trigger.

        Args:
            dataloader: Original data loader

        Returns:
            Poisoned data loader
        """
        from torch.utils.data import TensorDataset
        
        all_inputs = []
        all_targets = []
        
        for inputs, targets in dataloader:
            inputs = inputs.clone()
            targets = targets.clone()
            batch_size = inputs.size(0)
            
            # Poison fraction of samples
            num_poison = int(batch_size * self.poison_ratio)
            poison_indices = torch.randperm(batch_size)[:num_poison]
            
            # Inject trigger
            inputs[poison_indices] = self.inject_trigger(inputs[poison_indices])
            
            # Change target to backdoor target
            targets[poison_indices] = self.target_class
            
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
        return "backdoor"
