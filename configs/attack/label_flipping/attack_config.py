"""
Attack configuration for label flipping attacks on Federated Learning.

This module defines the configuration parameters for implementing various
label flipping attacks in a federated learning setting.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class AttackConfig:
    """
    Configuration for label flipping attacks.

    Attributes:
        attack_type: Type of label flipping attack ('random', 'targeted', 'inverse')
        flip_rate: Probability of flipping labels (0.0 to 1.0)
        malicious_client_indices: List of client IDs that are malicious (None for random selection)
        malicious_fraction: Fraction of clients that are malicious (if indices not specified)
        attack_start_round: Round number when attack begins (for delayed attacks)
        random_seed: Random seed for reproducibility
    """

    attack_type: Literal["random", "targeted", "inverse"]
    flip_rate: float = 0.5
    malicious_client_indices: Optional[list[int]] = None
    malicious_fraction: float = 0.2
    attack_start_round: int = 1
    random_seed: int = 42

    def __post_init__(self):
        """Validate attack configuration parameters."""
        if not 0.0 <= self.flip_rate <= 1.0:
            raise ValueError(f"flip_rate must be between 0.0 and 1.0, got {self.flip_rate}")

        if not 0.0 <= self.malicious_fraction <= 1.0:
            raise ValueError(f"malicious_fraction must be between 0.0 and 1.0, got {self.malicious_fraction}")

        if self.attack_start_round < 1:
            raise ValueError(f"attack_start_round must be >= 1, got {self.attack_start_round}")

        if self.attack_type not in ["random", "targeted", "inverse"]:
            raise ValueError(f"attack_type must be 'random', 'targeted', or 'inverse', got {self.attack_type}")

        # For inverse attack, flip_rate should be 1.0 (all labels flipped)
        if self.attack_type == "inverse" and self.flip_rate != 1.0:
            raise ValueError("Inverse attack requires flip_rate=1.0 (all labels flipped)")


def get_attack_configs() -> dict[str, AttackConfig]:
    """
    Get predefined attack configurations for experiments.

    Returns:
        Dictionary mapping configuration names to AttackConfig objects
    """
    return {
        "baseline": AttackConfig(attack_type="random", flip_rate=0.0, malicious_fraction=0.0),
        "random_10": AttackConfig(attack_type="random", flip_rate=0.3, malicious_fraction=0.1),
        "random_20": AttackConfig(attack_type="random", flip_rate=0.3, malicious_fraction=0.2),
        "random_30": AttackConfig(attack_type="random", flip_rate=0.3, malicious_fraction=0.3),
        "random_50": AttackConfig(attack_type="random", flip_rate=0.3, malicious_fraction=0.5),
        "targeted_20": AttackConfig(attack_type="targeted", flip_rate=0.5, malicious_fraction=0.2),
        "inverse_20": AttackConfig(attack_type="inverse", flip_rate=1.0, malicious_fraction=0.2),
        "delayed_20": AttackConfig(attack_type="random", flip_rate=0.3, malicious_fraction=0.2, attack_start_round=20),
    }
