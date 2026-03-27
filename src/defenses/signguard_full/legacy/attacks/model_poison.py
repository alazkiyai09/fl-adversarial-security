"""Model poisoning attack implementation."""

import torch
from typing import Optional, Literal

from src.defenses.signguard_full.legacy.attacks.base import Attack
from src.defenses.signguard_full.legacy.core.types import ModelUpdate


class ModelPoisonAttack(Attack):
    """Model poisoning attack.

    Directly modifies the model update to manipulate the global model.
    Includes scaling and sign-flip variants.
    """

    def __init__(
        self,
        attack_type: Literal["scaling", "sign_flip", "gaussian"] = "scaling",
        magnitude: float = -5.0,
        target_layers: Optional[list[str]] = None,
        sign_flip_ratio: float = 1.0,
    ):
        """Initialize model poisoning attack.

        Args:
            attack_type: Type of attack ("scaling", "sign_flip", "gaussian")
            magnitude: Scaling factor or attack magnitude
            target_layers: Specific layers to attack (None = all)
            sign_flip_ratio: Fraction of parameters to flip (for sign_flip)
        """
        self.attack_type = attack_type
        self.magnitude = magnitude
        self.target_layers = target_layers
        self.sign_flip_ratio = sign_flip_ratio

    def execute(
        self,
        client_id: str,
        global_model: dict[str, torch.Tensor],
        train_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> ModelUpdate:
        """Execute model poisoning attack.

        Args:
            client_id: Client identifier
            global_model: Current global model
            train_loader: Not used for model poisoning

        Returns:
            Malicious model update
        """
        malicious_params = {}
        
        for param_name, param_value in global_model.items():
            # Check if this layer should be attacked
            if self.target_layers is not None:
                if not any(layer in param_name for layer in self.target_layers):
                    # Skip this layer
                    malicious_params[param_name] = torch.zeros_like(param_value)
                    continue
            
            if self.attack_type == "scaling":
                # Scale the parameters
                malicious_params[param_name] = param_value * (self.magnitude - 1.0)
                
            elif self.attack_type == "sign_flip":
                # Flip signs of parameters
                if self.sign_flip_ratio >= 1.0:
                    # Flip all
                    malicious_params[param_name] = -param_value * abs(self.magnitude)
                else:
                    # Flip fraction
                    num_params = param_value.numel()
                    num_flip = int(num_params * self.sign_flip_ratio)
                    
                    flat_param = param_value.flatten()
                    flip_indices = torch.randperm(num_params)[:num_flip]
                    
                    flat_param[flip_indices] *= -1
                    malicious_params[param_name] = flat_param.view_as(param_value) * abs(self.magnitude)
                    
            elif self.attack_type == "gaussian":
                # Add Gaussian noise
                noise = torch.randn_like(param_value) * self.magnitude
                malicious_params[param_name] = noise
                
            else:
                raise ValueError(f"Unknown attack type: {self.attack_type}")
        
        return ModelUpdate(
            client_id=client_id,
            round_num=0,
            parameters=malicious_params,
            num_samples=100,
            metrics={"loss": 3.0, "accuracy": 0.2},
        )

    def get_name(self) -> str:
        """Get attack name."""
        return f"model_poison_{self.attack_type}"
