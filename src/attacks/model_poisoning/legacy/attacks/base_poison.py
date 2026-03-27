"""
Base class for model poisoning attacks.

All poisoning strategies must inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
import torch


class ModelPoisoningAttack(ABC):
    """
    Abstract base class for model poisoning attacks.

    Model poisoning attacks directly manipulate the model updates (gradients/weights)
    sent by malicious clients to the server, unlike data poisoning which manipulates
    training samples.
    """

    def __init__(self, attack_name: str):
        """
        Initialize the attack strategy.

        Args:
            attack_name: Name identifier for this attack
        """
        self.attack_name = attack_name
        self.attack_count = 0  # Track number of attacks executed

    @abstractmethod
    def poison_update(
        self,
        parameters: np.ndarray,
        layer_info: Dict[str, torch.Size]
    ) -> np.ndarray:
        """
        Apply poisoning to the model update.

        Args:
            parameters: Honest gradient/weight update as flattened numpy array
            layer_info: Dictionary mapping layer names to their shapes

        Returns:
            Poisoned parameters as flattened numpy array
        """
        pass

    def should_attack(
        self,
        current_round: int,
        timing_strategy: str = "continuous",
        frequency: int = 1,
        start_round: int = 0
    ) -> bool:
        """
        Determine if attack should be executed based on timing strategy.

        Args:
            current_round: Current federated learning round
            timing_strategy: "continuous", "intermittent", or "late_stage"
            frequency: Attack every N rounds (for intermittent)
            start_round: Start attacking after this round (for late_stage)

        Returns:
            True if attack should be executed, False otherwise
        """
        if timing_strategy == "continuous":
            return True
        elif timing_strategy == "intermittent":
            return current_round % frequency == 0
        elif timing_strategy == "late_stage":
            return current_round >= start_round
        else:
            raise ValueError(f"Unknown timing strategy: {timing_strategy}")

    def get_attack_info(self) -> Dict:
        """
        Get information about this attack for logging.

        Returns:
            Dictionary with attack metadata
        """
        return {
            "attack_name": self.attack_name,
            "attacks_executed": self.attack_count
        }

    def reset(self):
        """Reset attack counter."""
        self.attack_count = 0
