"""
Main Anomaly Detector for SignGuard

Orchestrates all 4 detection factors with configurable weights.
Provides online anomaly detection for federated learning.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

from .factors import AnomalyFactors, AnomalyThreshold
from .statistics import OnlineStatistics


class AnomalyDetector:
    """
    Main anomaly detection module for SignGuard.

    Features:
    - Multi-factor anomaly detection
    - Online statistics tracking
    - Adaptive thresholding
    - Per-client reputation tracking
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize AnomalyDetector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()

        # Extract detection config
        detection_config = self.config.get('detection', {})

        # Initialize factors
        self.factors = AnomalyFactors(detection_config)

        # Factor weights
        self.factor_weights = detection_config.get('factor_weights', {
            'l2_magnitude': 0.30,
            'directional_consistency': 0.25,
            'layer_wise': 0.25,
            'temporal_consistency': 0.20
        })

        # Anomaly threshold
        threshold_method = 'adaptive' if detection_config.get('use_adaptive_threshold', True) else 'percentile'
        self.threshold = AnomalyThreshold(
            method=threshold_method,
            percentile=detection_config.get('l2_threshold_percentile', 95)
        )

        # Online statistics
        self.stats = OnlineStatistics(
            ema_alpha=0.1,
            temporal_window=detection_config.get('temporal_window', 5)
        )

        # Detection history for analysis
        self.detection_history: List[Dict] = []

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'detection': {
                'factor_weights': {
                    'l2_magnitude': 0.30,
                    'directional_consistency': 0.25,
                    'layer_wise': 0.25,
                    'temporal_consistency': 0.20
                },
                'l2_threshold_percentile': 95,
                'cosine_similarity_threshold': 0.7,
                'layer_anomaly_threshold': 2.0,
                'temporal_threshold': 3.0,
                'temporal_window': 5,
                'anomaly_threshold': 0.5,
                'use_adaptive_threshold': True
            }
        }

    def detect_anomalies(self,
                         updates: Dict[str, List[np.ndarray]],
                         global_update: Optional[List[np.ndarray]] = None,
                         layer_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Detect anomalies across all client updates.

        Args:
            updates: Dictionary mapping client_id to model update (list of arrays)
            global_update: Optional global model update for directional analysis
            layer_names: Optional names for model layers

        Returns:
            Dictionary mapping client_id to anomaly results with:
                - scores: Individual factor scores
                - combined: Combined anomaly score
                - is_anomalous: Boolean flag
        """
        # Convert to list format
        client_ids = list(updates.keys())
        update_list = [updates[cid] for cid in client_ids]

        # Compute statistics
        for client_id, update in zip(client_ids, update_list):
            self.stats.update_client(client_id, update)

        # Compute median update (robust to outliers)
        median_update = self.stats.compute_median_update(update_list)

        # Use global update or fall back to median
        reference_direction = global_update if global_update is not None else median_update

        # Detect anomalies for each client
        results = {}
        all_combined_scores = []

        for client_id, update in zip(client_ids, update_list):
            # Get client history
            client_history = self.stats.get_client_history(client_id)

            # Compute all factors
            factor_scores = self.factors.compute_combined_anomaly(
                update=update,
                median_update=median_update,
                global_direction=reference_direction,
                client_id=client_id,
                client_history=client_history,
                layer_names=layer_names,
                factor_weights=self.factor_weights
            )

            # Extract combined score
            combined_score = factor_scores['combined']
            all_combined_scores.append(combined_score)

            # Check if anomalous
            threshold = self.config['detection'].get('anomaly_threshold', 0.5)
            is_anomalous = combined_score > threshold

            results[client_id] = {
                'scores': factor_scores,
                'combined': combined_score,
                'is_anomalous': is_anomalous
            }

        # Update adaptive threshold
        self.threshold.update(all_combined_scores)

        # Store in history
        self.detection_history.append({
            'round': len(self.detection_history),
            'results': results,
            'threshold': self.threshold.get_threshold()
        })

        return results

    def compute_single_anomaly(self,
                               update: List[np.ndarray],
                               client_id: str,
                               median_update: List[np.ndarray],
                               global_direction: List[np.ndarray],
                               layer_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute anomaly score for a single update.

        Convenience method for incremental detection.

        Args:
            update: Client's model update
            client_id: Client identifier
            median_update: Median update across clients
            global_direction: Global update direction
            layer_names: Optional layer names

        Returns:
            Dictionary with factor scores and combined score
        """
        # Update statistics
        self.stats.update_client(client_id, update)

        # Get client history
        client_history = self.stats.get_client_history(client_id)

        # Compute anomaly
        factor_scores = self.factors.compute_combined_anomaly(
            update=update,
            median_update=median_update,
            global_direction=global_direction,
            client_id=client_id,
            client_history=client_history,
            layer_names=layer_names,
            factor_weights=self.factor_weights
        )

        return factor_scores

    def get_anomaly_threshold(self) -> float:
        """Get current anomaly threshold."""
        return self.threshold.get_threshold()

    def is_anomalous(self, score: float) -> bool:
        """Check if score is anomalous."""
        return score > self.get_anomaly_threshold()

    def get_statistics(self) -> OnlineStatistics:
        """Get online statistics object."""
        return self.stats

    def get_detection_history(self) -> List[Dict]:
        """Get detection history."""
        return self.detection_history

    def get_anomaly_summary(self, round_results: Dict[str, Dict]) -> Dict:
        """
        Get summary statistics for a round of detection.

        Args:
            round_results: Results from detect_anomalies()

        Returns:
            Summary dictionary with statistics
        """
        combined_scores = [r['combined'] for r in round_results.values()]
        num_anomalous = sum(1 for r in round_results.values() if r['is_anomalous'])

        return {
            'num_clients': len(round_results),
            'num_anomalous': num_anomalous,
            'anomaly_rate': num_anomalous / len(round_results) if round_results else 0.0,
            'mean_score': float(np.mean(combined_scores)) if combined_scores else 0.0,
            'std_score': float(np.std(combined_scores)) if combined_scores else 0.0,
            'min_score': float(np.min(combined_scores)) if combined_scores else 0.0,
            'max_score': float(np.max(combined_scores)) if combined_scores else 0.0,
            'median_score': float(np.median(combined_scores)) if combined_scores else 0.0,
            'threshold': self.get_anomaly_threshold()
        }

    def reset(self) -> None:
        """Reset detector state."""
        self.stats.reset_all()
        self.threshold.reset()
        self.detection_history.clear()

    def set_factor_weights(self, weights: Dict[str, float]) -> None:
        """
        Update factor weights.

        Args:
            weights: Dictionary mapping factor names to weights
        """
        # Normalize weights
        total_weight = sum(weights.values())
        self.factor_weights = {k: v / total_weight for k, v in weights.items()}

    def get_factor_weights(self) -> Dict[str, float]:
        """Get current factor weights."""
        return self.factor_weights.copy()

    def get_top_anomalous_clients(self, results: Dict[str, Dict],
                                   top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k most anomalous clients.

        Args:
            results: Results from detect_anomalies()
            top_k: Number of top clients to return

        Returns:
            List of (client_id, anomaly_score) tuples
        """
        # Sort by anomaly score
        sorted_clients = sorted(
            results.items(),
            key=lambda x: x[1]['combined'],
            reverse=True
        )

        return [(cid, r['combined']) for cid, r in sorted_clients[:top_k]]


def compute_anomaly_score(update: np.ndarray,
                          global_update: np.ndarray,
                          client_id: str,
                          stats: OnlineStatistics,
                          factor_weights: Optional[Dict[str, float]] = None) -> float:
    """
    Standalone function to compute anomaly score.

    Convenience function for simple use cases.

    Args:
        update: Client's model update
        global_update: Global model update
        client_id: Client identifier
        stats: Online statistics object
        factor_weights: Optional factor weights

    Returns:
        Combined anomaly score
    """
    detector = AnomalyDetector()
    if factor_weights:
        detector.set_factor_weights(factor_weights)

    # Get median and history from stats
    client_history = stats.get_client_history(client_id)

    # Compute using factors
    factor_scores = detector.factors.compute_combined_anomaly(
        update=update,
        median_update=global_update,  # Use global as median
        global_direction=global_update,
        client_id=client_id,
        client_history=client_history
    )

    return factor_scores['combined']
