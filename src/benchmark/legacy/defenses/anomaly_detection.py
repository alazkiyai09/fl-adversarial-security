"""
Anomaly Detection defense for identifying malicious client updates.

Uses statistical methods to detect and reject anomalous updates that deviate
significantly from the majority.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from numpy.typing import NDArray
from scipy import stats

from .base import BaseDefense


class AnomalyDetectionDefense(BaseDefense):
    """
    Statistical anomaly detection for federated learning.

    Identifies malicious clients by detecting outliers in parameter updates
    using z-score, IQR, or isolation forest methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anomaly Detection defense.

        Args:
            config: Configuration with keys:
                - method: Detection method ('zscore', 'iqr', 'isolation_forest', 'mahalanobis')
                - threshold: Detection threshold (default: 3.0 for zscore)
                - use_median: Use median instead of mean for robustness (default: True)
                - min_clients: Minimum clients required (default: 3)
        """
        super().__init__(config)
        self.method = config.get("method", "zscore")
        self.threshold = config.get("threshold", 3.0)
        self.use_median = config.get("use_median", True)
        self.min_clients = config.get("min_clients", 3)

        # Detection metrics
        self._detected_indices: Optional[List[int]] = None
        self._anomaly_scores: Optional[NDArray] = None
        self._false_positive_rate: float = 0.0

    def reset_state(self) -> None:
        """Reset defense state between experiments."""
        self._detected_indices = None
        self._anomaly_scores = None
        self._false_positive_rate = 0.0

    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate updates with anomaly detection and rejection.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters from non-anomalous clients
        """
        if len(updates) == 0:
            return np.array([])

        if len(updates) < self.min_clients:
            # Not enough clients, fall back to simple mean
            params = self._extract_updates(updates)
            return np.mean(params, axis=0)

        params = self._extract_updates(updates)
        n_clients = len(params)

        # Detect anomalies
        anomaly_mask = self._detect_anomalies(params)
        self._detected_indices = [
            client_id for (client_id, _), is_anomaly in zip(updates, anomaly_mask)
            if is_anomaly
        ]

        # Filter out anomalous updates
        clean_indices = ~anomaly_mask
        clean_params = params[clean_indices]

        if len(clean_params) == 0:
            # All detected as anomalous, fall back to median
            return np.median(params, axis=0)

        # Aggregate clean updates
        if self.use_median:
            return np.median(clean_params, axis=0)
        else:
            return np.mean(clean_params, axis=0)

    def _detect_anomalies(self, params: NDArray) -> NDArray:
        """
        Detect anomalous parameter updates.

        Args:
            params: Array of shape (n_clients, n_params)

        Returns:
            Boolean array marking anomalous clients
        """
        if self.method == "zscore":
            return self._zscore_detection(params)
        elif self.method == "iqr":
            return self._iqr_detection(params)
        elif self.method == "isolation_forest":
            return self._isolation_forest_detection(params)
        elif self.method == "mahalanobis":
            return self._mahalanobis_detection(params)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

    def _zscore_detection(self, params: NDArray) -> NDArray:
        """
        Z-score based anomaly detection.

        Args:
            params: Array of shape (n_clients, n_params)

        Returns:
            Boolean array marking anomalous clients
        """
        n_clients = len(params)

        # Compute centroid (median for robustness)
        centroid = np.median(params, axis=0)

        # Compute z-scores for each client
        z_scores = np.zeros(n_clients)
        for i in range(n_clients):
            # Distance from centroid
            distance = np.linalg.norm(params[i] - centroid)
            # Normalize by std of distances
            z_scores[i] = distance / (np.std(params) + 1e-8)

        # Flag anomalies
        anomaly_mask = np.abs(z_scores) > self.threshold
        self._anomaly_scores = z_scores

        return anomaly_mask

    def _iqr_detection(self, params: NDArray) -> NDArray:
        """
        Interquartile range (IQR) based anomaly detection.

        Args:
            params: Array of shape (n_clients, n_params)

        Returns:
            Boolean array marking anomalous clients
        """
        # Compute distances from median
        centroid = np.median(params, axis=0)
        distances = np.array([np.linalg.norm(p - centroid) for p in params])

        # Compute IQR
        q25 = np.percentile(distances, 25)
        q75 = np.percentile(distances, 75)
        iqr = q75 - q25

        # Define bounds
        lower_bound = q25 - self.threshold * iqr
        upper_bound = q75 + self.threshold * iqr

        # Flag anomalies
        anomaly_mask = (distances < lower_bound) | (distances > upper_bound)
        self._anomaly_scores = distances

        return anomaly_mask

    def _isolation_forest_detection(self, params: NDArray) -> NDArray:
        """
        Isolation Forest based anomaly detection.

        Args:
            params: Array of shape (n_clients, n_params)

        Returns:
            Boolean array marking anomalous clients
        """
        try:
            from sklearn.ensemble import IsolationForest

            # Fit isolation forest
            clf = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            predictions = clf.fit_predict(params)

            # Isolation forest returns 1 for normal, -1 for anomaly
            anomaly_mask = predictions == -1
            self._anomaly_scores = clf.score_samples(params)

            return anomaly_mask

        except ImportError:
            # Fallback to z-score if sklearn not available
            return self._zscore_detection(params)

    def _mahalanobis_detection(self, params: NDArray) -> NDArray:
        """
        Mahalanobis distance based anomaly detection.

        Args:
            params: Array of shape (n_clients, n_params)

        Returns:
            Boolean array marking anomalous clients
        """
        # Compute robust covariance and centroid
        centroid = np.median(params, axis=0)

        try:
            # Use Minimum Covariance Determinant for robust estimation
            from sklearn.covariance import MinCovDet

            mcd = MinCovDet(support_fraction=min(0.9, (len(params) - 1) / len(params)))
            mcd.fit(params)
            robust_cov = mcd.covariance_
        except ImportError:
            # Fallback to regular covariance
            robust_cov = np.cov(params.T) + np.eye(params.shape[1]) * 1e-8

        # Compute Mahalanobis distances
        try:
            inv_cov = np.linalg.inv(robust_cov)
        except np.linalg.LinAlgError:
            inv_cov = np.eye(robust_cov.shape[0])

        distances = np.zeros(len(params))
        for i in range(len(params)):
            diff = params[i] - centroid
            distances[i] = np.sqrt(diff @ inv_cov @ diff.T)

        # Use chi-squared distribution for threshold
        # Degrees of freedom = number of parameters
        df = params.shape[1]
        critical_value = stats.chi2.ppf(0.95, df) if df < 100 else self.threshold

        anomaly_mask = distances > critical_value
        self._anomaly_scores = distances

        return anomaly_mask

    def get_detection_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get detection metrics from current round.

        Returns:
            Dictionary with detection metrics
        """
        if self._detected_indices is None:
            return None

        n_detected = len(self._detected_indices)

        metrics = {
            "num_detected": n_detected,
            "detection_rate": n_detected / 1.0 if n_detected > 0 else 0.0,  # Placeholder
        }

        if self._anomaly_scores is not None:
            metrics.update({
                "max_anomaly_score": float(np.max(self._anomaly_scores)),
                "min_anomaly_score": float(np.min(self._anomaly_scores)),
                "mean_anomaly_score": float(np.mean(self._anomaly_scores)),
                "std_anomaly_score": float(np.std(self._anomaly_scores)),
            })

        return metrics


class ClusteringDefense(BaseDefense):
    """
    Clustering-based defense that groups similar updates and selects the largest cluster.

    Based on the idea that honest clients will form a large cluster, while
    malicious clients will form smaller clusters or outliers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clustering defense.

        Args:
            config: Configuration with keys:
                - n_clusters: Number of clusters (default: 2)
                - distance_metric: Distance metric ('euclidean', 'cosine')
        """
        super().__init__(config)
        self.n_clusters = config.get("n_clusters", 2)
        self.distance_metric = config.get("distance_metric", "euclidean")
        self._cluster_labels: Optional[NDArray] = None
        self._largest_cluster_id: Optional[int] = None

    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate updates using clustering-based selection.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters from largest cluster
        """
        if len(updates) == 0:
            return np.array([])

        params = self._extract_updates(updates)
        n_clients = len(params)

        if n_clients <= 2:
            # Too few clients, just average
            return np.mean(params, axis=0)

        # Perform clustering
        cluster_labels = self._cluster_updates(params)
        self._cluster_labels = cluster_labels

        # Find largest cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        largest_cluster_id = unique_labels[np.argmax(counts)]
        self._largest_cluster_id = largest_cluster_id

        # Average updates from largest cluster
        largest_cluster_mask = cluster_labels == largest_cluster_id
        largest_cluster_params = params[largest_cluster_mask]

        return np.mean(largest_cluster_params, axis=0)

    def _cluster_updates(self, params: NDArray) -> NDArray:
        """
        Cluster client updates.

        Args:
            params: Array of shape (n_clients, n_params)

        Returns:
            Cluster labels for each client
        """
        try:
            from sklearn.cluster import KMeans

            # Use K-means clustering
            n_clusters = min(self.n_clusters, len(params))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(params)

            return labels

        except ImportError:
            # Fallback: simple distance-based clustering
            return self._simple_clustering(params)

    def _simple_clustering(self, params: NDArray) -> NDArray:
        """
        Simple clustering based on distance to centroids.

        Args:
            params: Array of shape (n_clients, n_params)

        Returns:
            Cluster labels
        """
        n_clusters = min(self.n_clusters, len(params))

        # Initialize centroids randomly
        centroid_indices = np.random.choice(len(params), n_clusters, replace=False)
        centroids = params[centroid_indices]

        # Assign to nearest centroid
        labels = np.zeros(len(params), dtype=int)

        for i, param in enumerate(params):
            if self.distance_metric == "cosine":
                distances = [
                    1 - np.dot(param, centroid) / (np.linalg.norm(param) * np.linalg.norm(centroid) + 1e-8)
                    for centroid in centroids
                ]
            else:
                distances = [np.linalg.norm(param - centroid) for centroid in centroids]

            labels[i] = np.argmin(distances)

        return labels

    def get_detection_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get detection metrics.

        Returns:
            Dictionary with cluster information
        """
        if self._cluster_labels is None:
            return None

        unique_labels, counts = np.unique(self._cluster_labels, return_counts=True)

        return {
            "num_clusters": len(unique_labels),
            "largest_cluster_size": int(np.max(counts)),
            "largest_cluster_id": int(self._largest_cluster_id) if self._largest_cluster_id is not None else -1,
            "cluster_sizes": {int(label): int(count) for label, count in zip(unique_labels, counts)},
        }
