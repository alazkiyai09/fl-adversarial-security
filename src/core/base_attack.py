"""
Base Attack Interface for Federated Learning Adversarial Attacks.

This module provides a unified abstract base class for all adversarial attack
implementations in the federated learning context. All attack implementations
should inherit from BaseAttack to ensure consistent interfaces across projects.

Example:
    >>> class MyAttack(BaseAttack):
    ...     def attack(self, data, model, round_info=None):
    ...         # Implement attack logic
    ...         return poisoned_data, attack_metadata
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from enum import Enum


class AttackTiming(Enum):
    """Attack timing strategies."""
    CONTINUOUS = "continuous"  # Attack every round
    EARLY = "early"  # Attack only early rounds
    LATE = "late"  # Attack only late rounds
    ALTERNATING = "alternating"  # Attack every other round
    ONCE = "once"  # Attack only once


@dataclass
class AttackConfig:
    """
    Standard configuration for all adversarial attacks.

    Attributes:
        attack_type: Type of attack (e.g., "label_flip", "backdoor", "model_poisoning")
        attack_rate: Fraction of data/clients to attack (0.0 to 1.0)
        attack_strength: Magnitude of attack (for scaling-based attacks)
        target_class: Target class for targeted attacks
        source_class: Source class for source-target attacks
        timing_strategy: When to apply the attack
        random_seed: Random seed for reproducibility
        max_attacks: Maximum number of attacks to perform
    """
    attack_type: str
    attack_rate: float = 0.3
    attack_strength: float = 1.0
    target_class: Optional[int] = None
    source_class: Optional[int] = None
    timing_strategy: str = AttackTiming.CONTINUOUS.value
    random_seed: int = 42
    max_attacks: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.attack_rate <= 1:
            raise ValueError(f"attack_rate must be in [0, 1], got {self.attack_rate}")
        if self.attack_strength < 0:
            raise ValueError(f"attack_strength must be >= 0, got {self.attack_strength}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'attack_type': self.attack_type,
            'attack_rate': self.attack_rate,
            'attack_strength': self.attack_strength,
            'target_class': self.target_class,
            'source_class': self.source_class,
            'timing_strategy': self.timing_strategy,
            'random_seed': self.random_seed,
            'max_attacks': self.max_attacks,
        }


@dataclass
class AttackMetadata:
    """
    Metadata returned by attacks for tracking and analysis.

    Attributes:
        attack_type: Type of attack performed
        num_affected: Number of samples/clients affected
        round_num: Round when attack occurred
        success: Whether attack was successful
        details: Additional attack-specific details
    """
    attack_type: str
    num_affected: int
    round_num: int
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):
    """
    Abstract base class for federated learning adversarial attacks.

    All attack implementations should inherit from this class and implement
    the required methods to ensure consistent interfaces.

    Attributes:
        config: Attack configuration
        attack_count: Number of attacks performed so far
        attack_history: History of attack metadata

    Example:
        >>> class LabelFlipAttack(BaseAttack):
        ...     def attack(self, data, model, round_info=None):
        ...         # Poison labels
        ...         return poisoned_data, AttackMetadata(...)
    """

    def __init__(self, config: AttackConfig):
        """
        Initialize attack with configuration.

        Args:
            config: Attack configuration
        """
        self.config = config
        self.attack_count = 0
        self.attack_history: list[AttackMetadata] = []

    @abstractmethod
    def attack(
        self,
        data: Any,
        model: Any,
        round_info: Optional[Dict] = None
    ) -> Tuple[Any, AttackMetadata]:
        """
        Apply attack to data/model.

        Args:
            data: Input data (can be features, labels, gradients, etc.)
            model: Model being attacked (can be model or weights)
            round_info: Optional round information (round_num, client_id, etc.)

        Returns:
            Tuple of (poisoned_data, attack_metadata)
        """
        pass

    def should_attack(self, round_num: int) -> bool:
        """
        Determine if attack should be applied in this round.

        Args:
            round_num: Current round number

        Returns:
            True if attack should be applied
        """
        # Check max attacks limit
        if self.config.max_attacks is not None:
            if self.attack_count >= self.config.max_attacks:
                return False

        # Apply timing strategy
        strategy = AttackTiming(self.config.timing_strategy)

        if strategy == AttackTiming.CONTINUOUS:
            return True
        elif strategy == AttackTiming.EARLY:
            return round_num < 10  # Attack first 10 rounds
        elif strategy == AttackTiming.LATE:
            return round_num >= 10  # Attack after round 10
        elif strategy == AttackTiming.ALTERNATING:
            return round_num % 2 == 0
        elif strategy == AttackTiming.ONCE:
            return self.attack_count == 0
        else:
            return True

    def get_attack_config(self) -> Dict[str, Any]:
        """
        Get attack configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.to_dict()

    def get_attack_info(self) -> Dict[str, Any]:
        """
        Get attack statistics and history.

        Returns:
            Dictionary with attack statistics
        """
        return {
            'attack_type': self.config.attack_type,
            'attack_count': self.attack_count,
            'attack_history': [
                {
                    'round': m.round_num,
                    'num_affected': m.num_affected,
                    'success': m.success,
                    'details': m.details
                }
                for m in self.attack_history
            ]
        }

    def reset(self):
        """Reset attack state and history."""
        self.attack_count = 0
        self.attack_history.clear()

    def _record_attack(self, metadata: AttackMetadata):
        """
        Record attack in history.

        Args:
            metadata: Attack metadata to record
        """
        self.attack_count += 1
        self.attack_history.append(metadata)


class AgnosticAttack(BaseAttack):
    """
    Base class for model-agnostic attacks (affect any model).

    These attacks work by poisoning data/labels without needing
    to know the model architecture.
    """

    pass


class ModelSpecificAttack(BaseAttack):
    """
    Base class for model-specific attacks (require model knowledge).

    These attacks work by directly manipulating model weights or gradients.
    """

    pass


class ClientSideAttack(BaseAttack):
    """
    Base class for client-side attacks (performed by malicious clients).

    These attacks are executed during local training.
    """

    pass


class ServerSideAttack(BaseAttack):
    """
    Base class for server-side attacks (performed by malicious server).

    These attacks are executed during aggregation.
    """

    pass
