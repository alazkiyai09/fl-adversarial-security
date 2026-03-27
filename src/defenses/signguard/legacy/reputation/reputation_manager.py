"""
Reputation Manager for SignGuard

Implements dynamic reputation system with decay.
Updates reputation based on anomaly scores.
"""

from typing import Dict, Optional, List
import numpy as np
from collections import defaultdict

from ..detection.anomaly_detector import AnomalyDetector


class ReputationManager:
    """
    Dynamic reputation system for federated learning clients.

    Features:
    - Exponential decay of historical reputation
    - Update based on anomaly scores
    - Probationary period for new clients
    - Bounds to prevent full exclusion
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize ReputationManager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()

        # Extract reputation config
        rep_config = self.config.get('reputation', {})

        self.initial_reputation = rep_config.get('initial_reputation', 0.5)
        self.min_reputation = rep_config.get('min_reputation', 0.01)
        self.max_reputation = rep_config.get('max_reputation', 1.0)
        self.decay_factor = rep_config.get('decay_factor', 0.9)
        self.probation_rounds = rep_config.get('probation_rounds', 5)
        self.probation_weight_multiplier = rep_config.get('probation_weight_multiplier', 0.5)

        # Track client reputations
        self.reputations: Dict[str, float] = {}
        self.client_rounds: Dict[str, int] = {}  # Rounds participated
        self.history: Dict[str, List[float]] = defaultdict(list)  # Reputation history

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'reputation': {
                'initial_reputation': 0.5,
                'min_reputation': 0.01,
                'max_reputation': 1.0,
                'decay_factor': 0.9,
                'probation_rounds': 5,
                'probation_weight_multiplier': 0.5
            }
        }

    def register_client(self, client_id: str) -> None:
        """
        Register a new client with initial reputation.

        Args:
            client_id: Client identifier
        """
        if client_id not in self.reputations:
            self.reputations[client_id] = self.initial_reputation
            self.client_rounds[client_id] = 0

    def update_reputation(self, client_id: str, anomaly_score: float) -> float:
        """
        Update reputation for a client based on anomaly score.

        Reputation update rule:
        R_{t+1} = α * R_t + (1-α) * (1 - anomaly_score)

        Where α is the decay factor.

        Args:
            client_id: Client identifier
            anomaly_score: Anomaly score in [0, 1] (0=benign, 1=malicious)

        Returns:
            Updated reputation value
        """
        # Register if new client
        self.register_client(client_id)

        # Get current reputation
        current_rep = self.reputations[client_id]

        # Compute new reputation
        # Low anomaly -> higher reputation
        # High anomaly -> lower reputation
        new_rep = (
            self.decay_factor * current_rep +
            (1 - self.decay_factor) * (1 - anomaly_score)
        )

        # Apply bounds
        new_rep = max(self.min_reputation, min(self.max_reputation, new_rep))

        # Store
        self.reputations[client_id] = new_rep
        self.client_rounds[client_id] += 1
        self.history[client_id].append(new_rep)

        return new_rep

    def batch_update(self, anomaly_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Update reputations for multiple clients.

        Args:
            anomaly_scores: Dictionary mapping client_id to anomaly score

        Returns:
            Dictionary of updated reputations
        """
        updated = {}

        for client_id, score in anomaly_scores.items():
            new_rep = self.update_reputation(client_id, score)
            updated[client_id] = new_rep

        return updated

    def get_reputation(self, client_id: str) -> float:
        """
        Get current reputation for a client.

        Args:
            client_id: Client identifier

        Returns:
            Reputation value (initial if not registered)
        """
        if client_id not in self.reputations:
            return self.initial_reputation
        return self.reputations[client_id]

    def get_all_reputations(self) -> Dict[str, float]:
        """Get all client reputations."""
        return self.reputations.copy()

    def get_effective_weight(self, client_id: str) -> float:
        """
        Get effective aggregation weight for a client.

        Accounts for probationary period.

        Args:
            client_id: Client identifier

        Returns:
            Effective weight (reputation * probation_multiplier if on probation)
        """
        rep = self.get_reputation(client_id)

        # Check if on probation
        if self.client_rounds.get(client_id, 0) < self.probation_rounds:
            return rep * self.probation_weight_multiplier

        return rep

    def get_all_weights(self) -> Dict[str, float]:
        """
        Get effective weights for all clients.

        Returns:
            Dictionary mapping client_id to effective weight
        """
        return {
            client_id: self.get_effective_weight(client_id)
            for client_id in self.reputations
        }

    def is_on_probation(self, client_id: str) -> bool:
        """
        Check if client is on probation.

        Args:
            client_id: Client identifier

        Returns:
            True if on probation, False otherwise
        """
        rounds = self.client_rounds.get(client_id, 0)
        return rounds < self.probation_rounds

    def get_reputation_history(self, client_id: str) -> List[float]:
        """
        Get reputation history for a client.

        Args:
            client_id: Client identifier

        Returns:
            List of historical reputation values
        """
        return self.history.get(client_id, []).copy()

    def get_reputation_stats(self) -> Dict[str, float]:
        """
        Get statistics about reputation distribution.

        Returns:
            Dictionary with statistics
        """
        reps = list(self.reputations.values())

        if len(reps) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }

        return {
            'mean': float(np.mean(reps)),
            'std': float(np.std(reps)),
            'min': float(np.min(reps)),
            'max': float(np.max(reps)),
            'median': float(np.median(reps))
        }

    def reset_client(self, client_id: str) -> None:
        """
        Reset reputation for a client.

        Args:
            client_id: Client identifier
        """
        if client_id in self.reputations:
            del self.reputations[client_id]
        if client_id in self.client_rounds:
            del self.client_rounds[client_id]
        if client_id in self.history:
            del self.history[client_id]

    def reset_all(self) -> None:
        """Reset all reputations."""
        self.reputations.clear()
        self.client_rounds.clear()
        self.history.clear()

    def get_low_reputation_clients(self, threshold: float = 0.2) -> List[str]:
        """
        Get clients with reputation below threshold.

        Args:
            threshold: Reputation threshold

        Returns:
            List of client IDs
        """
        return [
            client_id
            for client_id, rep in self.reputations.items()
            if rep < threshold
        ]

    def get_high_reputation_clients(self, threshold: float = 0.8) -> List[str]:
        """
        Get clients with reputation above threshold.

        Args:
            threshold: Reputation threshold

        Returns:
            List of client IDs
        """
        return [
            client_id
            for client_id, rep in self.reputations.items()
            if rep > threshold
        ]


def update_reputation(current_reputation: float,
                      anomaly_score: float,
                      decay_factor: float = 0.9) -> float:
    """
    Standalone function to update reputation.

    Args:
        current_reputation: Current reputation value
        anomaly_score: Anomaly score in [0, 1]
        decay_factor: Exponential decay factor (alpha)

    Returns:
        Updated reputation value
    """
    new_rep = (
        decay_factor * current_reputation +
        (1 - decay_factor) * (1 - anomaly_score)
    )
    return max(0.01, min(1.0, new_rep))
