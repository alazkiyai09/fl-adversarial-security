"""
Base detector interface for FL anomaly detection.
All detection methods must inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detectors in federated learning.

    All detectors must implement:
    - fit(): Learn normal behavior from baseline (honest) updates
    - compute_anomaly_score(): Return a continuous anomaly score
    - is_malicious(): Return binary decision based on threshold

    Design principles:
    - Unsupervised: Works without ground truth labels
    - Configurable: Thresholds adjustable for precision-recall tradeoff
    - Efficient: <100ms per client per round
    """

    def __init__(self, threshold: float = 3.0):
        """
        Initialize detector.

        Args:
            threshold: Detection threshold (higher = more conservative,
                      fewer false positives but more false negatives)
        """
        self.threshold = threshold
        self.is_fitted = False

    @abstractmethod
    def fit(self, baseline_updates: List[np.ndarray]) -> None:
        """
        Learn normal behavior from honest client updates.

        This method should compute any statistics needed for detection
        (e.g., mean, std, PCA components, cluster centroids).

        Args:
            baseline_updates: List of flattened model updates from honest clients

        Raises:
            ValueError: If baseline_updates is empty
        """
        if not baseline_updates:
            raise ValueError("baseline_updates cannot be empty")

        self.is_fitted = True

    @abstractmethod
    def compute_anomaly_score(self, update: np.ndarray, **kwargs) -> float:
        """
        Compute continuous anomaly score for a client update.

        Higher scores indicate more suspicious updates.

        Args:
            update: Flattened model update (numpy array)
            **kwargs: Additional detector-specific parameters
                     (e.g., global_model, client_id for historical detector)

        Returns:
            Anomaly score (non-negative float)

        Raises:
            RuntimeError: If detector has not been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before computing scores")
        return 0.0

    def is_malicious(self, update: np.ndarray, **kwargs) -> bool:
        """
        Binary decision: is the client malicious?

        Default implementation uses threshold on anomaly score.
        Override for custom decision logic.

        Args:
            update: Flattened model update
            **kwargs: Additional detector-specific parameters

        Returns:
            True if update is flagged as malicious, False otherwise
        """
        score = self.compute_anomaly_score(update, **kwargs)
        return score > self.threshold

    def reset(self) -> None:
        """Reset detector state (e.g., for new experiment)."""
        self.is_fitted = False

    def __repr__(self) -> str:
        """String representation of detector."""
        return f"{self.__class__.__name__}(threshold={self.threshold})"
