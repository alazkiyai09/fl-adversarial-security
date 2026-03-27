"""Multi-factor ensemble anomaly detector."""

import torch
from typing import List, Dict

from src.defenses.signguard_full.legacy.core.types import ModelUpdate, AnomalyScore
from src.defenses.signguard_full.legacy.detection.base import AnomalyDetector
from src.defenses.signguard_full.legacy.detection.magnitude_detector import L2NormDetector
from src.defenses.signguard_full.legacy.detection.direction_detector import CosineSimilarityDetector
from src.defenses.signguard_full.legacy.detection.score_detector import LossDeviationDetector


class EnsembleDetector(AnomalyDetector):
    """Multi-factor ensemble detector for anomaly detection.

    Combines multiple anomaly detectors (magnitude, direction, loss)
    using configurable weighting schemes.
    """

    def __init__(
        self,
        magnitude_weight: float = 0.4,
        direction_weight: float = 0.4,
        loss_weight: float = 0.2,
        anomaly_threshold: float = 0.7,
        ensemble_method: str = "weighted",
    ):
        """Initialize ensemble detector.

        Args:
            magnitude_weight: Weight for L2 norm detector
            direction_weight: Weight for cosine similarity detector
            loss_weight: Weight for loss deviation detector
            anomaly_threshold: Combined anomaly threshold
            ensemble_method: How to combine scores ('weighted', 'voting', 'max')
        """
        self.magnitude_weight = magnitude_weight
        self.direction_weight = direction_weight
        self.loss_weight = loss_weight
        self.anomaly_threshold = anomaly_threshold
        self.ensemble_method = ensemble_method

        # Validate weights sum to 1 (for weighted method)
        if ensemble_method == "weighted":
            total_weight = magnitude_weight + direction_weight + loss_weight
            if abs(total_weight - 1.0) > 1e-6:
                # Normalize weights
                self.magnitude_weight = magnitude_weight / total_weight
                self.direction_weight = direction_weight / total_weight
                self.loss_weight = loss_weight / total_weight

        # Initialize individual detectors
        self.magnitude_detector = L2NormDetector()
        self.direction_detector = CosineSimilarityDetector()
        self.loss_detector = LossDeviationDetector()

    def compute_score(
        self,
        update: ModelUpdate,
        global_model: dict[str, torch.Tensor],
        client_history: List[ModelUpdate] | None = None,
    ) -> float:
        """Compute combined anomaly score.

        Args:
            update: Client's model update
            global_model: Current global model
            client_history: Optional historical updates from this client

        Returns:
            Combined anomaly score in [0, 1]
        """
        # Compute individual scores
        magnitude_score = self.magnitude_detector.compute_score(
            update, global_model, client_history
        )
        direction_score = self.direction_detector.compute_score(
            update, global_model, client_history
        )
        loss_score = self.loss_detector.compute_score(
            update, global_model, client_history
        )

        # Combine scores
        if self.ensemble_method == "weighted":
            combined_score = (
                self.magnitude_weight * magnitude_score +
                self.direction_weight * direction_score +
                self.loss_weight * loss_score
            )
        elif self.ensemble_method == "voting":
            # Majority voting: each detector votes if score > threshold
            votes = sum([
                magnitude_score > self.anomaly_threshold,
                direction_score > self.anomaly_threshold,
                loss_score > self.anomaly_threshold,
            ])
            combined_score = votes / 3.0
        elif self.ensemble_method == "max":
            # Take maximum score
            combined_score = max(magnitude_score, direction_score, loss_score)
        elif self.ensemble_method == "min":
            # Take minimum score
            combined_score = min(magnitude_score, direction_score, loss_score)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        return combined_score

    def compute_anomaly_score(
        self,
        update: ModelUpdate,
        global_model: dict[str, torch.Tensor],
        client_history: List[ModelUpdate] | None = None,
    ) -> AnomalyScore:
        """Compute detailed multi-factor anomaly score.

        Args:
            update: Client's model update
            global_model: Current global model
            client_history: Optional historical updates from this client

        Returns:
            AnomalyScore with all component scores
        """
        # Compute individual scores
        magnitude_score = self.magnitude_detector.compute_score(
            update, global_model, client_history
        )
        direction_score = self.direction_detector.compute_score(
            update, global_model, client_history
        )
        loss_score = self.loss_detector.compute_score(
            update, global_model, client_history
        )

        # Compute combined score
        combined_score = self.compute_score(update, global_model, client_history)

        return AnomalyScore(
            magnitude_score=magnitude_score,
            direction_score=direction_score,
            loss_score=loss_score,
            combined_score=combined_score,
        )

    def is_anomalous(self, anomaly_score: AnomalyScore) -> bool:
        """Check if update exceeds anomaly threshold.

        Args:
            anomaly_score: Computed anomaly score

        Returns:
            True if update is anomalous
        """
        return anomaly_score.combined_score > self.anomaly_threshold

    def update_statistics(
        self,
        updates: List[ModelUpdate],
        global_model: dict[str, torch.Tensor],
    ) -> None:
        """Update all detectors' statistics.

        Args:
            updates: List of model updates
            global_model: Current global model
        """
        self.magnitude_detector.update_statistics(updates, global_model)
        self.direction_detector.update_statistics(updates, global_model)
        self.loss_detector.update_statistics(updates, global_model)

    def get_detector_statistics(self) -> Dict[str, Dict]:
        """Get statistics from all detectors.

        Returns:
            Dictionary mapping detector name -> statistics
        """
        return {
            "magnitude": self.magnitude_detector.get_statistics(),
            "direction": self.direction_detector.get_statistics(),
            "loss": self.loss_detector.get_statistics(),
        }
