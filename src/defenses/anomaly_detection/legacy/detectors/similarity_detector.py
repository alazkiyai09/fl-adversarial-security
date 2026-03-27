"""
Similarity-based anomaly detector using cosine similarity.
Detects clients whose update direction differs from normal clients.
"""

from typing import List, Literal
import numpy as np

from .base_detector import BaseDetector


class SimilarityDetector(BaseDetector):
    """
    Detects malicious clients by analyzing direction of model updates.

    Assumption: Malicious clients send updates in different directions
    from honest clients (e.g., opposite direction for backdoor attacks).

    Detection: Flag updates with low cosine similarity to reference.
    Reference can be:
    - global_model: Compare to global model parameters
    - median_update: Compare to median client update
    - mean_update: Compare to mean client update
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        comparison_target: Literal["global_model", "median_update", "mean_update"] = "global_model"
    ):
        """
        Initialize similarity detector.

        Args:
            similarity_threshold: Minimum cosine similarity (0-1)
                                 Lower = more sensitive
            comparison_target: What to compare against
        """
        # Threshold is inverted for anomaly scoring (low similarity = high score)
        # We convert to: anomaly score when similarity < threshold
        super().__init__(threshold=similarity_threshold)
        self.comparison_target = comparison_target

        # Reference update learned from baseline
        self.reference_update: np.ndarray = None

    def fit(self, baseline_updates: List[np.ndarray]) -> None:
        """
        Learn reference update from honest clients.

        Args:
            baseline_updates: List of flattened updates from honest clients
        """
        super().fit(baseline_updates)

        # Stack updates for computation
        updates_matrix = np.vstack(baseline_updates)

        if self.comparison_target == "median_update":
            self.reference_update = np.median(updates_matrix, axis=0)
        elif self.comparison_target == "mean_update":
            self.reference_update = np.mean(updates_matrix, axis=0)
        else:  # global_model
            # For global model, reference is the mean (centroid)
            # In practice, you'd use the actual global model parameters
            self.reference_update = np.mean(updates_matrix, axis=0)

    def compute_anomaly_score(
        self,
        update: np.ndarray,
        global_model: np.ndarray = None,
        **kwargs
    ) -> float:
        """
        Compute anomaly score based on cosine similarity deviation.

        Score = max(0, threshold - similarity)
        Higher score = more dissimilar = more anomalous

        Args:
            update: Flattened model update
            global_model: Optional global model parameters (if comparison_target="global_model")
            **kwargs: Additional parameters

        Returns:
            Anomaly score (non-negative, higher = more anomalous)
        """
        # Choose reference
        if self.comparison_target == "global_model" and global_model is not None:
            reference = global_model
        else:
            reference = self.reference_update

        # Compute cosine similarity
        similarity = self._cosine_similarity(update, reference)

        # Anomaly score: how far below threshold
        # If similarity >= threshold, score = 0 (not anomalous)
        # If similarity < threshold, score = threshold - similarity (anomalous)
        score = max(0.0, self.threshold - similarity)

        return float(score)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            a, b: Flattened parameter arrays

        Returns:
            Cosine similarity in [-1, 1]
        """
        # Flatten if needed
        a_flat = a.flatten()
        b_flat = b.flatten()

        # Compute cosine similarity
        dot_product = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        # Clip to [-1, 1] (handle numerical errors)
        similarity = max(-1.0, min(1.0, similarity))

        return float(similarity)

    def get_similarity(self, update: np.ndarray, **kwargs) -> float:
        """
        Get raw cosine similarity (for analysis).

        Args:
            update: Flattened model update
            **kwargs: May include global_model parameter

        Returns:
            Cosine similarity in [-1, 1]
        """
        global_model = kwargs.get('global_model', None)

        if self.comparison_target == "global_model" and global_model is not None:
            reference = global_model
        else:
            reference = self.reference_update

        return self._cosine_similarity(update, reference)
