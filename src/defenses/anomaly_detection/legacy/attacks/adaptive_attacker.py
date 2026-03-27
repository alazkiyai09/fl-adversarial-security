"""
Adaptive attacker that evades anomaly detection.
Simulates sophisticated attackers who adapt to detection thresholds.
"""

from typing import Dict, Optional
import numpy as np


class AdaptiveAttacker:
    """
    Simulates adaptive attackers who modify their strategy to evade detection.

    Attack strategies:
    1. threshold_aware: Scale attack to stay below detection threshold
    2. gradual: Start with small attacks, gradually increase magnitude
    3. camouflage: Mimic honest client behavior (add noise similar to normal)
    4. label_flipping: Flip labels without changing model significantly
    """

    def __init__(
        self,
        strategy: str = "threshold_aware",
        detection_threshold: float = 3.0,
        target_layer: Optional[str] = None
    ):
        """
        Initialize adaptive attacker.

        Args:
            strategy: Attack strategy
            detection_threshold: Known or estimated detection threshold
            target_layer: Specific layer to target (for backdoor attacks)
        """
        self.strategy = strategy
        self.detection_threshold = detection_threshold
        self.target_layer = target_layer
        self.round_count = 0

    def generate_attack(
        self,
        honest_update: np.ndarray,
        attack_magnitude: float = 1.0,
        attack_direction: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate malicious update adapted to evade detection.

        Args:
            honest_update: Baseline honest update to modify
            attack_magnitude: Base strength of attack
            attack_direction: Optional direction for attack (e.g., backdoor vector)

        Returns:
            Malicious update crafted to evade detection
        """
        self.round_count += 1

        if self.strategy == "threshold_aware":
            return self._threshold_aware_attack(
                honest_update, attack_magnitude, attack_direction
            )

        elif self.strategy == "gradual":
            return self._gradual_attack(
                honest_update, attack_magnitude, attack_direction
            )

        elif self.strategy == "camouflage":
            return self._camouflage_attack(
                honest_update, attack_magnitude, attack_direction
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _threshold_aware_attack(
        self,
        honest_update: np.ndarray,
        attack_magnitude: float,
        attack_direction: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Scale attack to stay just below detection threshold.

        Strategy: Compute L2 norm, ensure it's within threshold * std of honest.
        """
        honest_norm = np.linalg.norm(honest_update)

        # Add attack in direction (or random if none specified)
        if attack_direction is not None:
            attack_dir = attack_direction / np.linalg.norm(attack_direction)
        else:
            attack_dir = np.random.randn(*honest_update.shape)
            attack_dir = attack_dir / np.linalg.norm(attack_dir)

        # Scale attack to stay below threshold
        # Target: honest_norm + attack is just below threshold
        max_delta = self.detection_threshold * honest_norm * 0.9  # 90% of threshold
        attack_delta = attack_dir * min(attack_magnitude, max_delta - honest_norm)

        return honest_update + attack_delta

    def _gradual_attack(
        self,
        honest_update: np.ndarray,
        attack_magnitude: float,
        attack_direction: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Gradually increase attack magnitude over rounds.

        Strategy: Start small (hard to detect), increase slowly.
        """
        # Gradual scaling: 0.1 -> 0.2 -> ... -> 1.0 over 10 rounds
        scale_factor = min(1.0, self.round_count / 10.0)

        if attack_direction is not None:
            attack_dir = attack_direction / np.linalg.norm(attack_direction)
        else:
            attack_dir = np.random.randn(*honest_update.shape)
            attack_dir = attack_dir / np.linalg.norm(attack_dir)

        attack_delta = attack_dir * attack_magnitude * scale_factor

        return honest_update + attack_delta

    def _camouflage_attack(
        self,
        honest_update: np.ndarray,
        attack_magnitude: float,
        attack_direction: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Camouflage attack to mimic honest behavior.

        Strategy: Add attack component + noise similar to honest variance.
        """
        # Base attack
        if attack_direction is not None:
            attack = attack_direction * attack_magnitude
        else:
            attack = np.random.randn(*honest_update.shape) * attack_magnitude

        # Add noise similar to honest update variance
        honest_std = np.std(honest_update)
        camouflage_noise = np.random.randn(*honest_update.shape) * honest_std

        return honest_update + attack + camouflage_noise

    def reset_rounds(self) -> None:
        """Reset round counter (for new experiment)."""
        self.round_count = 0
