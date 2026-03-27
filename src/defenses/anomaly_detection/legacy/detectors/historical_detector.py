"""
Historical behavior detector.
Tracks client reputation over multiple FL rounds using exponential smoothing.
"""

from typing import List, Dict
import numpy as np

from .base_detector import BaseDetector


class HistoricalDetector(BaseDetector):
    """
    Detects malicious clients by tracking their behavior over time.

    Assumption: Malicious clients consistently deviate from normal behavior,
    while honest clients may have occasional anomalies.

    Detection:
    1. Maintain exponential moving average (EMA) of scores per client
    2. Flag clients whose current score deviates significantly from their EMA
    3. Reputation = EMA of scores (lower = more trusted)

    Benefits:
    - Reduces false positives for honest clients with rare anomalies
    - Detects consistently malicious clients even with subtle attacks
    - Adapts to concept drift (client behavior changes over time)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        threshold: float = 2.0,
        warmup_rounds: int = 5
    ):
        """
        Initialize historical detector.

        Args:
            alpha: EMA smoothing factor (0-1).
                   Higher = more weight on recent behavior
                   Lower = more weight on historical behavior
            threshold: Z-score threshold for deviation from reputation
            warmup_rounds: Number of rounds before using historical detector
                           (need baseline reputation first)
        """
        super().__init__(threshold=threshold)
        self.alpha = alpha
        self.warmup_rounds = warmup_rounds

        # Client reputation tracking
        self.reputations: Dict[str, float] = {}  # client_id -> EMA of scores
        self.client_rounds: Dict[str, int] = {}  # client_id -> rounds seen

    def fit(self, baseline_updates: List[np.ndarray] = None) -> None:
        """
        No fitting needed for historical detector (stateless initialization).

        Args:
            baseline_updates: Ignored (kept for interface compatibility)
        """
        self.is_fitted = True

    def compute_anomaly_score(
        self,
        update: np.ndarray,
        client_id: str,
        **kwargs
    ) -> float:
        """
        Compute anomaly score based on client reputation.

        Score = deviation from expected behavior (reputation)
        Higher score = more unexpected given client's history

        Args:
            update: Flattened model update
            client_id: Unique client identifier
            **kwargs: Additional parameters

        Returns:
            Anomaly score (higher = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before computing scores")

        # Update rounds counter
        if client_id not in self.client_rounds:
            self.client_rounds[client_id] = 0
        self.client_rounds[client_id] += 1

        # Get current reputation (or initialize)
        if client_id not in self.reputations:
            self.reputations[client_id] = 0.0  # Start neutral

        # During warmup, return 0 (not enough history)
        if self.client_rounds[client_id] <= self.warmup_rounds:
            return 0.0

        # Compute current deviation from reputation
        # (In practice, you'd compute score from another detector first)
        # Here we use L2 norm deviation as proxy
        current_norm = np.linalg.norm(update)
        expected_norm = self.reputations[client_id]

        # Z-score style deviation
        deviation = abs(current_norm - expected_norm) / (expected_norm + 1e-8)

        return float(deviation)

    def update_reputation(self, client_id: str, score: float) -> None:
        """
        Update client reputation using exponential moving average.

        Reputation_{t} = alpha * score_t + (1 - alpha) * Reputation_{t-1}

        Args:
            client_id: Unique client identifier
            score: Current anomaly score (e.g., from magnitude detector)
        """
        if client_id not in self.reputations:
            self.reputations[client_id] = score
        else:
            old_reputation = self.reputations[client_id]
            new_reputation = self.alpha * score + (1 - self.alpha) * old_reputation
            self.reputations[client_id] = new_reputation

    def get_reputation(self, client_id: str) -> float:
        """
        Get client's current reputation.

        Args:
            client_id: Unique client identifier

        Returns:
            Reputation score (lower = more trusted)
        """
        return self.reputations.get(client_id, 0.0)

    def is_malicious(self, update: np.ndarray, client_id: str, **kwargs) -> bool:
        """
        Binary decision: is client malicious?

        Considers both current deviation AND reputation.
        Client flagged if:
        - Current deviation > threshold, AND
        - Reputation is high (consistently anomalous)

        Args:
            update: Flattened model update
            client_id: Unique client identifier
            **kwargs: Additional parameters

        Returns:
            True if client is flagged as malicious
        """
        score = self.compute_anomaly_score(update, client_id)
        reputation = self.get_reputation(client_id)

        # Flag if both current and historical are anomalous
        # (reputation > threshold means consistently high scores)
        return score > self.threshold and reputation > self.threshold

    def reset_client(self, client_id: str) -> None:
        """
        Reset tracking for specific client (e.g., if they rejoin).

        Args:
            client_id: Unique client identifier
        """
        if client_id in self.reputations:
            del self.reputations[client_id]
        if client_id in self.client_rounds:
            del self.client_rounds[client_id]

    def reset(self) -> None:
        """Reset all client tracking."""
        super().reset()
        self.reputations.clear()
        self.client_rounds.clear()

    def get_all_reputations(self) -> Dict[str, float]:
        """
        Get all client reputations (for analysis).

        Returns:
            Dictionary mapping client_id -> reputation
        """
        return self.reputations.copy()
