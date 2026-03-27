"""
Attack detection using L2 norms and cosine similarity.

Monitors client updates to identify potential model poisoning attacks.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine


class AttackDetector:
    """
    Detect model poisoning attacks through statistical analysis.

    Uses two main detection mechanisms:
    1. L2 norm analysis: Flag updates with unusually large magnitude
    2. Cosine similarity: Flag updates with negative correlation to majority
    """

    def __init__(
        self,
        l2_norm_threshold: float = 10.0,
        cosine_similarity_threshold: float = -0.5,
        std_threshold: float = 3.0
    ):
        """
        Initialize attack detector.

        Args:
            l2_norm_threshold: Flag updates with L2 norm > mean + std * threshold
            cosine_similarity_threshold: Flag updates with cosine similarity < threshold
            std_threshold: Number of standard deviations for outlier detection
        """
        self.l2_norm_threshold = l2_norm_threshold
        self.cosine_threshold = cosine_similarity_threshold
        self.std_threshold = std_threshold

        # Detection history
        self.detection_history = []

    def detect_anomalies(
        self,
        client_updates: List[np.ndarray],
        client_ids: List[int] = None
    ) -> Dict:
        """
        Detect anomalous client updates using multiple metrics.

        Args:
            client_updates: List of flattened parameter updates
            client_ids: Optional list of client identifiers

        Returns:
            Dictionary with detection results
        """
        if client_ids is None:
            client_ids = list(range(len(client_updates)))

        num_clients = len(client_updates)
        if num_clients == 0:
            return {"suspicious_clients": [], "detection_details": {}}

        # Compute L2 norms
        l2_norms = [np.linalg.norm(update) for update in client_updates]

        # Compute pairwise cosine similarities
        cosine_matrix = self._compute_cosine_matrix(client_updates)

        # Detect L2 norm outliers
        l2_outliers = self._detect_l2_outliers(l2_norms, client_ids)

        # Detect cosine similarity outliers (negative correlation)
        cosine_outliers = self._detect_cosine_outliers(
            cosine_matrix,
            client_ids
        )

        # Combine detections
        suspicious_clients = list(set(l2_outliers + cosine_outliers))
        suspicious_clients.sort()

        detection_details = {
            "l2_norms": {cid: l2_norm for cid, l2_norm in zip(client_ids, l2_norms)},
            "l2_outliers": l2_outliers,
            "cosine_outliers": cosine_outliers,
            "cosine_matrix": cosine_matrix.tolist() if cosine_matrix is not None else None
        }

        result = {
            "suspicious_clients": suspicious_clients,
            "detection_details": detection_details
        }

        self.detection_history.append(result)

        return result

    def _compute_cosine_matrix(
        self,
        updates: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            updates: List of parameter updates

        Returns:
            Cosine similarity matrix
        """
        n = len(updates)
        cosine_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    cosine_matrix[i, j] = 1.0
                else:
                    # Compute cosine similarity: 1 - cosine_distance
                    cosine_matrix[i, j] = 1 - cosine(updates[i], updates[j])

        return cosine_matrix

    def _detect_l2_outliers(
        self,
        l2_norms: List[float],
        client_ids: List[int]
    ) -> List[int]:
        """
        Detect clients with unusually large L2 norms.

        Args:
            l2_norms: List of L2 norms
            client_ids: List of client identifiers

        Returns:
            List of suspicious client IDs
        """
        outliers = []

        mean_l2 = np.mean(l2_norms)
        std_l2 = np.std(l2_norms)

        threshold = mean_l2 + self.std_threshold * std_l2

        for client_id, l2 in zip(client_ids, l2_norms):
            if l2 > threshold:
                outliers.append(client_id)

        return outliers

    def _detect_cosine_outliers(
        self,
        cosine_matrix: np.ndarray,
        client_ids: List[int]
    ) -> List[int]:
        """
        Detect clients with negative correlation to majority.

        Args:
            cosine_matrix: Pairwise cosine similarity matrix
            client_ids: List of client identifiers

        Returns:
            List of suspicious client IDs
        """
        outliers = []
        n = len(client_ids)

        # Compute average cosine similarity for each client
        avg_similarities = []
        for i in range(n):
            # Exclude self-similarity (diagonal)
            row = np.concatenate([cosine_matrix[i, :i], cosine_matrix[i, i+1:]])
            avg_sim = np.mean(row) if len(row) > 0 else 1.0
            avg_similarities.append(avg_sim)

        # Flag clients with negative average similarity
        for client_id, avg_sim in zip(client_ids, avg_similarities):
            if avg_sim < self.cosine_threshold:
                outliers.append(client_id)

        return outliers

    def compute_l2_norm(self, parameters: np.ndarray) -> float:
        """
        Compute L2 norm of parameter vector.

        Args:
            parameters: Flattened parameter array

        Returns:
            L2 norm
        """
        return float(np.linalg.norm(parameters))

    def compute_cosine_similarity(
        self,
        update_a: np.ndarray,
        update_b: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two updates.

        Args:
            update_a: First parameter update
            update_b: Second parameter update

        Returns:
            Cosine similarity in range [-1, 1]
        """
        return float(1 - cosine(update_a, update_b))

    def get_detection_statistics(self) -> Dict:
        """
        Get summary statistics across all detections.

        Returns:
            Dictionary with detection statistics
        """
        if not self.detection_history:
            return {"total_detections": 0}

        total_detections = len(self.detection_history)

        # Count total suspicious clients across all rounds
        all_suspicious = []
        for detection in self.detection_history:
            all_suspicious.extend(detection["suspicious_clients"])

        unique_suspicious = list(set(all_suspicious))

        return {
            "total_detections": total_detections,
            "unique_suspicious_clients": unique_suspicious,
            "total_suspicious_detections": len(all_suspicious),
            "avg_suspicious_per_round": len(all_suspicious) / total_detections if total_detections > 0 else 0
        }
