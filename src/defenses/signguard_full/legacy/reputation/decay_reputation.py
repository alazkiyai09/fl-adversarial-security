"""Time-decay reputation system."""

from src.defenses.signguard_full.legacy.core.types import ReputationInfo
from src.defenses.signguard_full.legacy.reputation.base import ReputationSystem
from typing import Dict


class DecayReputationSystem(ReputationSystem):
    """Time-decay reputation system with honesty bonus and penalty.

    Reputation updates based on anomaly scores and signature verification.
    """

    def __init__(
        self,
        initial_reputation: float = 0.5,
        decay_rate: float = 0.05,
        honesty_bonus: float = 0.1,
        penalty_factor: float = 0.5,
        min_reputation: float = 0.0,
        max_reputation: float = 1.0,
    ):
        """Initialize decay reputation system.

        Args:
            initial_reputation: Starting reputation for new clients
            decay_rate: Exponential decay rate per round
            honesty_bonus: Bonus for low anomaly scores
            penalty_factor: Penalty multiplier for high anomaly scores
            min_reputation: Minimum reputation bound
            max_reputation: Maximum reputation bound
        """
        self.initial_reputation = initial_reputation
        self.decay_rate = decay_rate
        self.honesty_bonus = honesty_bonus
        self.penalty_factor = penalty_factor
        self.min_reputation = min_reputation
        self.max_reputation = max_reputation
        self.reputations: Dict[str, ReputationInfo] = {}

    def initialize_client(self, client_id: str) -> None:
        """Initialize reputation for new client.

        Args:
            client_id: Client identifier
        """
        if client_id not in self.reputations:
            self.reputations[client_id] = ReputationInfo(
                client_id=client_id,
                reputation=self.initial_reputation,
                num_contributions=0,
                last_update_round=0,
            )

    def update_reputation(
        self,
        client_id: str,
        anomaly_score: float,
        round_num: int,
        is_verified: bool = True,
    ) -> float:
        """Update client reputation based on behavior.

        Args:
            client_id: Client identifier
            anomaly_score: Detected anomaly score [0, 1]
            round_num: Current round number
            is_verified: Whether signature was verified

        Returns:
            Updated reputation value
        """
        # Initialize if new client
        if client_id not in self.reputations:
            self.initialize_client(client_id)

        info = self.reputations[client_id]

        # Apply time decay
        rounds_since_update = round_num - info.last_update_round
        decay = self.decay_rate**rounds_since_update
        new_rep = info.reputation * decay

        # Apply bonus or penalty based on anomaly score
        if is_verified:
            if anomaly_score < 0.3:  # Low anomaly = honest
                new_rep = min(
                    self.max_reputation, new_rep + self.honesty_bonus
                )
            elif anomaly_score > 0.7:  # High anomaly = suspicious
                penalty = anomaly_score * self.penalty_factor
                new_rep = max(self.min_reputation, new_rep - penalty)

        # Update info
        info.reputation = new_rep
        info.num_contributions += 1
        info.last_update_round = round_num
        info.add_detection_score(anomaly_score)

        return new_rep

    def get_reputation(self, client_id: str) -> float:
        """Get current reputation for client.

        Args:
            client_id: Client identifier

        Returns:
            Current reputation value
        """
        if client_id not in self.reputations:
            self.initialize_client(client_id)
        return self.reputations[client_id].reputation

    def get_all_reputations(self) -> Dict[str, float]:
        """Get reputations for all clients.

        Returns:
            Dict mapping client_id -> reputation
        """
        return {
            client_id: info.reputation for client_id, info in self.reputations.items()
        }
