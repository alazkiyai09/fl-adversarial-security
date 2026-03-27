"""
Clustering-based anomaly detector.
Uses DBSCAN and Isolation Forest for outlier detection.
"""

from typing import List, Literal
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from .base_detector import BaseDetector


class ClusteringDetector(BaseDetector):
    """
    Detects malicious clients using clustering algorithms.

    Methods:
    - isolation_forest: Tree-based outlier detection (fast, scalable)
    - dbscan: Density-based clustering (flags low-density points)

    Assumption: Honest clients form dense clusters in parameter space,
    while malicious clients are outliers (far from clusters).
    """

    def __init__(
        self,
        method: Literal["isolation_forest", "dbscan"] = "isolation_forest",
        contamination: float = 0.1,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 3
    ):
        """
        Initialize clustering detector.

        Args:
            method: Clustering algorithm
            contamination: Expected proportion of outliers (for Isolation Forest)
            dbscan_eps: Max distance between DBSCAN cluster samples
            dbscan_min_samples: Min samples for DBSCAN core point
        """
        super().__init__(threshold=0.5)  # Threshold for binary decision
        self.method = method
        self.contamination = contamination
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # Fitted models
        self.isolation_forest: IsolationForest = None
        self.dbscan: DBSCAN = None

        # For DBSCAN: compute cluster centroids
        self.cluster_centroids: np.ndarray = None
        self.cluster_labels_: np.ndarray = None

    def fit(self, baseline_updates: List[np.ndarray]) -> None:
        """
        Fit clustering model to honest client updates.

        Args:
            baseline_updates: List of flattened updates from honest clients
        """
        super().fit(baseline_updates)

        # Stack updates into matrix
        X = np.vstack(baseline_updates)

        if self.method == "isolation_forest":
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            self.isolation_forest.fit(X)

        elif self.method == "dbscan":
            self.dbscan = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                n_jobs=-1
            )
            self.dbscan.fit(X)

            # Compute centroids for each cluster
            unique_labels = set(self.dbscan.labels_)
            unique_labels.discard(-1)  # Remove noise points

            if unique_labels:
                centroids = []
                for label in unique_labels:
                    cluster_mask = self.dbscan.labels_ == label
                    centroid = np.mean(X[cluster_mask], axis=0)
                    centroids.append(centroid)
                self.cluster_centroids = np.vstack(centroids)
            else:
                # No clusters found, use mean as single centroid
                self.cluster_centroids = np.mean(X, axis=0).reshape(1, -1)

            self.cluster_labels_ = self.dbscan.labels_

    def compute_anomaly_score(self, update: np.ndarray, **kwargs) -> float:
        """
        Compute anomaly score based on clustering.

        For Isolation Forest: Returns negative outlier score (higher = more anomalous)
        For DBSCAN: Returns distance to nearest cluster centroid

        Args:
            update: Flattened model update

        Returns:
            Anomaly score (higher = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before computing scores")

        if self.method == "isolation_forest":
            # Isolation Forest: negative score = more anomalous
            # Convert to positive: score = -1 * raw_score
            raw_score = self.isolation_forest.score_samples(update.reshape(1, -1))[0]
            score = -1.0 * raw_score  # Higher = more anomalous

            # Normalize roughly to [0, 1] range
            score = max(0, score)

        elif self.method == "dbscan":
            # Distance to nearest cluster centroid
            if self.cluster_centroids is None:
                score = 0.0
            else:
                distances = np.linalg.norm(
                    self.cluster_centroids - update.reshape(1, -1),
                    axis=1
                )
                score = float(np.min(distances))

        return float(score)

    def is_malicious(self, update: np.ndarray, **kwargs) -> bool:
        """
        Binary decision: is client malicious?

        For Isolation Forest: Uses contamination-based threshold
        For DBSCAN: Flags if too far from any cluster

        Args:
            update: Flattened model update

        Returns:
            True if update is flagged as malicious
        """
        score = self.compute_anomaly_score(update)

        if self.method == "isolation_forest":
            # Isolation Forest has built-in threshold at 0
            # score > 0 means outlier
            return score > 0
        else:  # dbscan
            # Use configurable threshold
            return score > self.threshold

    def predict_cluster(self, update: np.ndarray) -> int:
        """
        Predict which cluster an update belongs to (DBSCAN only).

        Args:
            update: Flattened model update

        Returns:
            Cluster label (-1 for noise/outlier)
        """
        if self.method != "dbscan" or self.dbscan is None:
            return -1

        return self.dbscan.predict([update])[0]

    def get_num_clusters(self) -> int:
        """
        Get number of clusters found by DBSCAN.

        Returns:
            Number of clusters (excluding noise)
        """
        if self.method != "dbscan" or self.cluster_labels_ is None:
            return 0

        unique_labels = set(self.cluster_labels_)
        unique_labels.discard(-1)
        return len(unique_labels)
