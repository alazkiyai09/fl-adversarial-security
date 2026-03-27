"""Cosine similarity direction anomaly detector."""

import torch
import numpy as np
from typing import List

from src.defenses.signguard_full.legacy.detection.base import AnomalyDetector
from src.defenses.signguard_full.legacy.core.types import ModelUpdate
from src.defenses.signguard_full.legacy.utils.serialization import parameters_to_vector


class CosineSimilarityDetector(AnomalyDetector):
    """Cosine similarity-based anomaly detector.

    Detects anomalies by comparing the direction of parameter updates
    against the expected direction (mean or majority). Low cosine similarity
    indicates update is in a different direction, potentially malicious.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        similarity_metric: str = "cosine",
        use_robust_aggregation: bool = True,
        window_size: int = 10,
    ):
        """Initialize cosine similarity detector.

        Args:
            threshold: Anomaly threshold
            similarity_metric: Similarity metric ('cosine' or 'dot_product')
            use_robust_aggregation: Use median instead of mean for reference
            window_size: Window size for history
        """
        self.threshold = threshold
        self.similarity_metric = similarity_metric
        self.use_robust_aggregation = use_robust_aggregation
        self.window_size = window_size
        self.direction_history: List[torch.Tensor] = []

    def compute_score(
        self,
        update: ModelUpdate,
        global_model: dict[str, torch.Tensor],
        client_history: List[ModelUpdate] | None = None,
    ) -> float:
        """Compute cosine similarity-based anomaly score.

        Args:
            update: Client's model update
            global_model: Current global model
            client_history: Optional historical updates from this client

        Returns:
            Anomaly score in [0, 1], higher = more anomalous
        """
        # Compute update direction
        update_diff = self._compute_update_diff(update.parameters, global_model)
        update_vector = parameters_to_vector(update_diff)
        update_direction = update_vector / (torch.norm(update_vector) + 1e-10)

        # If we have history, compute similarity
        if self.direction_history:
            # Get reference direction
            ref_direction = self._get_reference_direction()
            
            # Compute similarity
            similarity = self._compute_similarity(update_direction, ref_direction)
            
            # Convert to anomaly score: low similarity = high anomaly
            # Cosine similarity in [-1, 1], we map to [0, 1]
            # similarity = 1 -> same direction (low anomaly)
            # similarity = 0 -> orthogonal (medium anomaly)
            # similarity = -1 -> opposite direction (high anomaly)
            anomaly_score = (1.0 - similarity) / 2.0
            return anomaly_score
        else:
            # No history, cannot detect direction anomaly
            return 0.0

    def update_statistics(
        self,
        updates: List[ModelUpdate],
        global_model: dict[str, torch.Tensor],
    ) -> None:
        """Update detector's internal statistics.

        Args:
            updates: List of model updates
            global_model: Current global model
        """
        directions = []
        for update in updates:
            update_diff = self._compute_update_diff(update.parameters, global_model)
            update_vector = parameters_to_vector(update_diff)
            update_direction = update_vector / (torch.norm(update_vector) + 1e-10)
            directions.append(update_direction)

        # Update history
        self.direction_history.extend(directions)
        if len(self.direction_history) > self.window_size:
            self.direction_history = self.direction_history[-self.window_size:]

    def _compute_update_diff(
        self,
        update_params: dict[str, torch.Tensor],
        global_params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute parameter difference.

        Args:
            update_params: Update parameters
            global_params: Global model parameters

        Returns:
            Parameter differences
        """
        return {
            name: update_params[name] - global_params[name]
            for name in update_params.keys()
        }

    def _get_reference_direction(self) -> torch.Tensor:
        """Get reference direction from history.

        Returns:
            Reference direction vector
        """
        if self.use_robust_aggregation:
            # Use median (element-wise) for robustness
            stacked = torch.stack(self.direction_history)
            ref_direction = torch.median(stacked, dim=0).values
        else:
            # Use mean
            ref_direction = torch.stack(self.direction_history).mean(dim=0)
        
        # Normalize
        ref_direction = ref_direction / (torch.norm(ref_direction) + 1e-10)
        return ref_direction

    def _compute_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score
        """
        if self.similarity_metric == "cosine":
            # Cosine similarity in [-1, 1]
            similarity = torch.dot(vec1, vec2).item()
            return max(-1.0, min(1.0, similarity))
        elif self.similarity_metric == "dot_product":
            # Dot product (not normalized)
            return torch.dot(vec1, vec2).item()
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

    def get_statistics(self) -> dict:
        """Get current detector statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.direction_history:
            return {
                "count": 0,
                "mean_cosine_similarity": 0.0,
            }

        # Compute pairwise similarities
        n = len(self.direction_history)
        similarities = []
        for i in range(n - 1):
            sim = self._compute_similarity(
                self.direction_history[i],
                self.direction_history[i + 1]
            )
            similarities.append(sim)

        return {
            "count": n,
            "mean_cosine_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "std_cosine_similarity": float(np.std(similarities)) if similarities else 0.0,
        }
