"""
Magnitude-based anomaly detector using L2 norms.
Detects outliers using z-score or IQR method.
"""

from typing import List, Literal
import numpy as np
from scipy import stats

from .base_detector import BaseDetector


class MagnitudeDetector(BaseDetector):
    """
    Detects malicious clients by analyzing L2 norm of model updates.

    Assumption: Malicious clients have significantly different update
    magnitudes compared to honest clients (either much larger or smaller).

    Methods:
    - zscore: Flag updates with norm z-score > threshold
    - iqr: Flag updates outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR]
    """

    def __init__(
        self,
        method: Literal["zscore", "iqr"] = "zscore",
        threshold: float = 3.0,
        iqr_multiplier: float = 1.5
    ):
        """
        Initialize magnitude detector.

        Args:
            method: Detection method ("zscore" or "iqr")
            threshold: Z-score threshold (for zscore method)
            iqr_multiplier: IQR multiplier (for iqr method, typically 1.5)
        """
        super().__init__(threshold=threshold)
        self.method = method
        self.iqr_multiplier = iqr_multiplier

        # Statistics learned from baseline
        self.mean_norm: float = 0.0
        self.std_norm: float = 0.0
        self.q1_norm: float = 0.0
        self.q3_norm: float = 0.0
        self.iqr_norm: float = 0.0

    def fit(self, baseline_updates: List[np.ndarray]) -> None:
        """
        Learn norm statistics from honest client updates.

        Args:
            baseline_updates: List of flattened updates from honest clients
        """
        super().fit(baseline_updates)

        # Compute L2 norms for all baseline updates
        norms = np.array([self._compute_l2_norm(update) for update in baseline_updates])

        # Compute statistics based on method
        if self.method == "zscore":
            self.mean_norm = np.mean(norms)
            self.std_norm = np.std(norms)
            if self.std_norm == 0:
                self.std_norm = 1e-8  # Avoid division by zero
        else:  # iqr
            self.q1_norm = np.percentile(norms, 25)
            self.q3_norm = np.percentile(norms, 75)
            self.iqr_norm = self.q3_norm - self.q1_norm
            if self.iqr_norm == 0:
                self.iqr_norm = 1e-8  # Avoid division by zero

    def compute_anomaly_score(self, update: np.ndarray, **kwargs) -> float:
        """
        Compute anomaly score based on L2 norm deviation.

        For zscore: Returns absolute z-score of norm
        For iqr: Returns distance from IQR boundary (normalized)

        Args:
            update: Flattened model update

        Returns:
            Anomaly score (higher = more anomalous)
        """
        norm = self._compute_l2_norm(update)

        if self.method == "zscore":
            # Absolute z-score
            score = abs((norm - self.mean_norm) / self.std_norm)
        else:  # iqr
            # Distance from acceptable range
            lower_bound = self.q1_norm - self.iqr_multiplier * self.iqr_norm
            upper_bound = self.q3_norm + self.iqr_multiplier * self.iqr_norm

            if norm < lower_bound:
                score = (lower_bound - norm) / self.iqr_norm
            elif norm > upper_bound:
                score = (norm - upper_bound) / self.iqr_norm
            else:
                score = 0.0

        return float(score)

    def _compute_l2_norm(self, update: np.ndarray) -> float:
        """
        Compute L2 norm of flattened update.

        Args:
            update: Flattened parameter array

        Returns:
            L2 norm (Euclidean norm)
        """
        return float(np.linalg.norm(update))

    def get_norm_statistics(self) -> dict:
        """
        Get learned norm statistics (useful for analysis/debugging).

        Returns:
            Dictionary with norm statistics
        """
        if self.method == "zscore":
            return {
                'method': 'zscore',
                'mean': self.mean_norm,
                'std': self.std_norm
            }
        else:
            return {
                'method': 'iqr',
                'q1': self.q1_norm,
                'q3': self.q3_norm,
                'iqr': self.iqr_norm,
                'multiplier': self.iqr_multiplier
            }
