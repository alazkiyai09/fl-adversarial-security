"""Base class for reputation systems."""

from abc import ABC, abstractmethod


class ReputationSystem(ABC):
    """Abstract base class for reputation systems."""

    @abstractmethod
    def update_reputation(
        self,
        client_id: str,
        anomaly_score: float,
        round_num: int,
        is_verified: bool = True,
    ) -> float:
        """Update reputation for a client."""
        pass

    @abstractmethod
    def get_reputation(self, client_id: str) -> float:
        """Get current reputation for a client."""
        pass
