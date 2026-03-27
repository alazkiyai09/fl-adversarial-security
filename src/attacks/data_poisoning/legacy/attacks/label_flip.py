"""
Label flipping attack implementations for Federated Learning.

This module implements various label flipping attacks that can be used
by malicious clients in a federated learning system.
"""

import numpy as np
from typing import Literal
from ..utils.poisoning_utils import (
    flip_labels,
    flip_fraud_to_legitimate,
    invert_labels,
    calculate_flip_statistics,
)


def random_flip(labels: np.ndarray, flip_prob: float) -> np.ndarray:
    """
    Random flip attack: Flip labels with probability p.

    This attack randomly flips labels (both 0->1 and 1->0) with a given probability.
    It creates noise in the training data and can degrade model performance.

    Args:
        labels: Original labels (binary: 0 or 1)
        flip_prob: Probability of flipping each label (0.0 to 1.0)

    Returns:
        Flipped labels as numpy array

    Example:
        >>> labels = np.array([0, 0, 1, 1, 0])
        >>> flipped = random_flip(labels, flip_prob=0.5)
        >>> # Approximately 50% of labels will be flipped
    """
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError(f"flip_prob must be between 0.0 and 1.0, got {flip_prob}")

    return flip_labels(labels, flip_prob)


def targeted_flip(labels: np.ndarray, flip_prob: float) -> np.ndarray:
    """
    Targeted flip attack: Flip only fraud labels (1) to legitimate (0).

    This is a stealthy attack where the adversary only flips fraud cases to
    legitimate. This is particularly harmful for fraud detection as it teaches
    the model to misclassify fraud as legitimate.

    Attack Goal: Reduce model's ability to detect fraud by poisoning the
    training data with mislabeled fraud cases.

    Args:
        labels: Original labels (binary: 0 or 1)
        flip_prob: Probability of flipping each fraud label to legitimate

    Returns:
        Flipped labels as numpy array

    Example:
        >>> labels = np.array([0, 0, 1, 1, 0])
        >>> flipped = targeted_flip(labels, flip_prob=0.5)
        >>> # Only fraud (1) labels may be flipped to 0
    """
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError(f"flip_prob must be between 0.0 and 1.0, got {flip_prob}")

    return flip_fraud_to_legitimate(labels, flip_prob)


def inverse_flip(labels: np.ndarray) -> np.ndarray:
    """
    Inverse flip attack: Flip all labels (0 -> 1, 1 -> 0).

    This is a severe attack that completely inverts all labels. It maximizes
    the impact on the model and can cause complete failure or lead the model
    to learn inverse classification.

    Attack Goal: Completely confuse the model by inverting all labels, causing
    maximum degradation in model performance.

    Args:
        labels: Original labels (binary: 0 or 1)

    Returns:
        Inverted labels as numpy array

    Example:
        >>> labels = np.array([0, 0, 1, 1, 0])
        >>> flipped = inverse_flip(labels)
        >>> # Returns: array([1, 1, 0, 0, 1])
    """
    return invert_labels(labels)


def apply_attack(
    labels: np.ndarray,
    attack_type: Literal["random", "targeted", "inverse"],
    flip_prob: float = 0.5,
    seed: int | None = None
) -> tuple[np.ndarray, dict]:
    """
    Apply a label flipping attack to the given labels.

    This is the main interface for applying attacks. It routes to the appropriate
    attack function based on the attack type and returns statistics about the attack.

    Args:
        labels: Original labels (binary: 0 or 1)
        attack_type: Type of attack ('random', 'targeted', or 'inverse')
        flip_prob: Probability of flipping (used for random and targeted attacks)
        seed: Random seed for reproducibility (None for random)

    Returns:
        Tuple of (poisoned_labels, attack_statistics)

    Example:
        >>> labels = np.array([0, 0, 1, 1, 0])
        >>> flipped, stats = apply_attack(labels, attack_type="targeted", flip_prob=0.5)
        >>> print(f"Flipped {stats['total_flips']} labels")
    """
    if seed is not None:
        np.random.seed(seed)

    original_labels = labels.copy()

    if attack_type == "random":
        poisoned_labels = random_flip(labels, flip_prob)
    elif attack_type == "targeted":
        poisoned_labels = targeted_flip(labels, flip_prob)
    elif attack_type == "inverse":
        poisoned_labels = inverse_flip(labels)
    else:
        raise ValueError(
            f"Unknown attack_type: {attack_type}. "
            "Must be 'random', 'targeted', or 'inverse'."
        )

    # Calculate attack statistics
    stats = calculate_flip_statistics(original_labels, poisoned_labels)

    return poisoned_labels, stats


class LabelFlipAttack:
    """
    Label flipping attack class for use in federated learning clients.

    This class encapsulates the attack logic and configuration, making it easy
    to integrate with Flower clients.
    """

    def __init__(
        self,
        attack_type: Literal["random", "targeted", "inverse"],
        flip_rate: float = 0.5,
        random_seed: int = 42
    ):
        """
        Initialize the label flipping attack.

        Args:
            attack_type: Type of attack ('random', 'targeted', or 'inverse')
            flip_rate: Probability/rate of flipping labels
            random_seed: Random seed for reproducibility
        """
        self.attack_type = attack_type
        self.flip_rate = flip_rate
        self.random_seed = random_seed

        # Validate attack configuration
        if self.attack_type == "inverse" and self.flip_rate != 1.0:
            raise ValueError("Inverse attack requires flip_rate=1.0")

        if not 0.0 <= self.flip_rate <= 1.0:
            raise ValueError(f"flip_rate must be between 0.0 and 1.0, got {self.flip_rate}")

    def poison_labels(self, labels: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Apply the label flipping attack to the given labels.

        Args:
            labels: Original labels (binary: 0 or 1)

        Returns:
            Tuple of (poisoned_labels, attack_statistics)
        """
        return apply_attack(
            labels,
            self.attack_type,
            self.flip_rate,
            self.random_seed
        )

    def __repr__(self) -> str:
        """String representation of the attack."""
        return (
            f"LabelFlipAttack(attack_type='{self.attack_type}', "
            f"flip_rate={self.flip_rate}, seed={self.random_seed})"
        )


# Convenience function for creating attacks
def create_attack(
    attack_type: Literal["random", "targeted", "inverse"],
    flip_rate: float = 0.5,
    random_seed: int = 42
) -> LabelFlipAttack:
    """
    Create a LabelFlipAttack instance.

    Args:
        attack_type: Type of attack ('random', 'targeted', or 'inverse')
        flip_rate: Probability/rate of flipping labels
        random_seed: Random seed for reproducibility

    Returns:
        LabelFlipAttack instance
    """
    return LabelFlipAttack(attack_type, flip_rate, random_seed)
