"""Loss deviation anomaly detector."""

import torch
import numpy as np
from typing import List

from src.defenses.signguard_full.legacy.detection.base import AnomalyDetector
from src.defenses.signguard_full.legacy.core.types import ModelUpdate


class LossDeviationDetector(AnomalyDetector):
    """Loss deviation-based anomaly detector.

    Detects anomalies by analyzing the loss values reported by clients.
    Abnormally high or low loss may indicate data poisoning or other attacks.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        use_iqr: bool = True,
        iqr_multiplier: float = 1.5,
        window_size: int = 10,
        loss_key: str = "loss",
    ):
        """Initialize loss deviation detector.

        Args:
            threshold: Fixed anomaly threshold (if not using IQR)
            use_iqr: Whether to use Interquartile Range for adaptive threshold
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5 for outliers)
            window_size: Window size for history
            loss_key: Key for loss value in metrics dict
        """
        self.threshold = threshold
        self.use_iqr = use_iqr
        self.iqr_multiplier = iqr_multiplier
        self.window_size = window_size
        self.loss_key = loss_key
        self.loss_history: List[float] = []

    def compute_score(
        self,
        update: ModelUpdate,
        global_model: dict[str, torch.Tensor],
        client_history: List[ModelUpdate] | None = None,
    ) -> float:
        """Compute loss deviation-based anomaly score.

        Args:
            update: Client's model update
            global_model: Current global model (unused)
            client_history: Optional historical updates from this client

        Returns:
            Anomaly score in [0, 1], higher = more anomalous
        """
        # Get loss from metrics
        loss = update.metrics.get(self.loss_key, None)
        
        if loss is None:
            # No loss reported, assume normal
            return 0.0
        
        # Normalize loss to anomaly score
        anomaly_score = self._normalize_loss(loss)
        
        return anomaly_score

    def update_statistics(self, updates: List[ModelUpdate], global_model: dict[str, torch.Tensor] | None = None) -> None:
        """Update detector's internal statistics.

        Args:
            updates: List of model updates
            global_model: Current global model (unused)
        """
        losses = []
        for update in updates:
            loss = update.metrics.get(self.loss_key)
            if loss is not None:
                losses.append(loss)

        # Update history
        self.loss_history.extend(losses)
        if len(self.loss_history) > self.window_size:
            self.loss_history = self.loss_history[-self.window_size:]

    def _normalize_loss(self, loss: float) -> float:
        """Normalize loss to [0, 1] anomaly score.

        Args:
            loss: Loss value

        Returns:
            Normalized anomaly score
        """
        if not self.loss_history:
            # No history, use simple scaling
            # Assume loss is typically in [0, 10]
            return min(loss / 10.0, 1.0)

        losses_array = np.array(self.loss_history)
        
        if self.use_iqr and len(self.loss_history) >= 4:
            # Use Interquartile Range
            q1 = np.percentile(losses_array, 25)
            q3 = np.percentile(losses_array, 75)
            iqr = q3 - q1
            
            # Bounds for outliers
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            
            # Compute deviation from bounds
            if loss < lower_bound:
                # Abnormally low loss
                deviation = (lower_bound - loss) / (abs(lower_bound) + 1e-10)
            elif loss > upper_bound:
                # Abnormally high loss
                deviation = (loss - upper_bound) / (abs(upper_bound) + 1e-10)
            else:
                # Within normal range
                deviation = 0.0
            
            # Convert to [0, 1]
            score = min(deviation, 1.0)
            return score
        
        else:
            # Use z-score based normalization
            mean = np.mean(losses_array)
            std = np.std(losses_array)
            
            if std < 1e-10:
                return 0.0
            
            # Z-score
            z_score = abs(loss - mean) / std
            
            # Convert to [0, 1] using exponential decay
            score = 1.0 - np.exp(-z_score ** 2 / 10)
            return min(score, 1.0)

    def get_statistics(self) -> dict:
        """Get current detector statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.loss_history:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        losses_array = np.array(self.loss_history)
        return {
            "count": len(losses_array),
            "mean": float(np.mean(losses_array)),
            "median": float(np.median(losses_array)),
            "std": float(np.std(losses_array)),
            "min": float(np.min(losses_array)),
            "max": float(np.max(losses_array)),
            "q25": float(np.percentile(losses_array, 25)),
            "q75": float(np.percentile(losses_array, 75)),
        }
