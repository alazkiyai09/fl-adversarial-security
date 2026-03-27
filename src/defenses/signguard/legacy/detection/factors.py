"""
Anomaly Detection Factors for SignGuard

Implements all 4 detection factors:
1. L2 Magnitude Anomaly
2. Directional Consistency
3. Layer-wise Analysis
4. Temporal Consistency
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import median_abs_deviation

from .statistics import OnlineStatistics


class AnomalyFactors:
    """
    Implements all 4 anomaly detection factors.

    Each factor computes an anomaly score in [0, 1] range:
    - 0: Normal/Benign
    - 1: Highly anomalous
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize AnomalyFactors.

        Args:
            config: Configuration dictionary with thresholds and weights
        """
        self.config = config or self._default_config()

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'l2_threshold_percentile': 95,
            'cosine_similarity_threshold': 0.7,
            'layer_anomaly_threshold': 2.0,  # MAD multiplier
            'temporal_threshold': 3.0,  # Std multiplier
            'temporal_window': 5
        }

    def compute_l2_anomaly(self, update: np.ndarray,
                           median_update: np.ndarray,
                           threshold: Optional[float] = None) -> float:
        """
        Factor 1: L2 Magnitude Anomaly

        Measures deviation in update magnitude from median.
        Large deviations may indicate model poisoning.

        Args:
            update: Client's model update
            median_update: Median update across clients
            threshold: Pre-computed threshold (auto-computed if None)

        Returns:
            Anomaly score in [0, 1]
        """
        # Flatten updates
        update_flat = self._flatten(update)
        median_flat = self._flatten(median_update)

        # Compute L2 norm difference
        l2_diff = np.linalg.norm(update_flat - median_flat)

        # Normalize by median norm (scale-invariant)
        median_norm = np.linalg.norm(median_flat) + 1e-8
        normalized_diff = l2_diff / median_norm

        # Convert to anomaly score using sigmoid
        # Using threshold as inflection point
        if threshold is None:
            threshold = self.config['l2_threshold_percentile'] / 100.0

        # Sigmoid mapping: higher diff -> higher anomaly
        anomaly_score = 1.0 / (1.0 + np.exp(-(normalized_diff - threshold) * 5))

        return float(anomaly_score)

    def compute_directional_anomaly(self, update: np.ndarray,
                                    global_direction: np.ndarray,
                                    threshold: Optional[float] = None) -> float:
        """
        Factor 2: Directional Consistency

        Measures cosine similarity to global update direction.
        Malicious clients often update in opposite direction.

        Args:
            update: Client's model update
            global_direction: Global/aggregated update direction
            threshold: Minimum cosine similarity (default from config)

        Returns:
            Anomaly score in [0, 1]
        """
        # Flatten updates
        update_flat = self._flatten(update)
        global_flat = self._flatten(global_direction)

        # Handle zero vectors
        update_norm = np.linalg.norm(update_flat)
        global_norm = np.linalg.norm(global_flat)

        if update_norm < 1e-8 or global_norm < 1e-8:
            return 0.5  # Neutral for zero updates

        # Compute cosine similarity (1 - cosine distance)
        cosine_sim = 1.0 - cosine(update_flat, global_flat)

        # Clamp to [-1, 1]
        cosine_sim = max(-1.0, min(1.0, cosine_sim))

        # Set threshold
        if threshold is None:
            threshold = self.config['cosine_similarity_threshold']

        # Convert to anomaly score
        # High similarity -> low anomaly
        # Low similarity -> high anomaly
        if cosine_sim >= threshold:
            # Below threshold: scale from [threshold, 1] to [0, 1]
            anomaly_score = (1.0 - cosine_sim) / (1.0 - threshold + 1e-8)
            anomaly_score = anomaly_score * 0.3  # Max 0.3 for above threshold
        else:
            # Below threshold: scale from [-1, threshold] to [0.3, 1]
            anomaly_score = 0.3 + 0.7 * (threshold - cosine_sim) / (threshold + 1.0 + 1e-8)

        return float(anomaly_score)

    def compute_layer_anomaly(self, update: np.ndarray,
                              median_update: np.ndarray,
                              layer_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Factor 3: Layer-wise Analysis

        Detects anomalies at layer level.
        Some attacks target specific layers (e.g., output layer for backdoor).

        Args:
            update: Client's model update (list of layer arrays)
            median_update: Median update (list of layer arrays)
            layer_names: Optional names for layers

        Returns:
            Dictionary mapping layer names to anomaly scores
        """
        # Ensure updates are lists
        if isinstance(update, np.ndarray):
            update = [update]
        if isinstance(median_update, np.ndarray):
            median_update = [median_update]

        # Generate layer names if not provided
        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(len(update))]

        layer_scores = {}
        mad_multiplier = self.config['layer_anomaly_threshold']

        for i, (layer_update, layer_median) in enumerate(zip(update, median_update)):
            # Compute L2 norm for this layer
            update_norm = np.linalg.norm(layer_update.flatten())
            median_norm = np.linalg.norm(layer_median.flatten())

            # Compute absolute deviation
            deviation = abs(update_norm - median_norm)

            # Use relative deviation (scale-invariant)
            relative_deviation = deviation / (median_norm + 1e-8)

            # Convert to anomaly score using MAD threshold
            # Assuming distribution is around 0, values > MAD * multiplier are anomalous
            anomaly_score = min(1.0, relative_deviation / mad_multiplier)

            layer_scores[layer_names[i]] = float(anomaly_score)

        return layer_scores

    def compute_temporal_anomaly(self, update: np.ndarray,
                                 client_history: List[np.ndarray]) -> float:
        """
        Factor 4: Temporal Consistency

        Compares current update to client's historical pattern.
        Sudden changes may indicate compromise or adaptive attacks.

        Args:
            update: Client's current model update
            client_history: List of client's past updates

        Returns:
            Anomaly score in [0, 1]
        """
        if len(client_history) == 0:
            return 0.0  # No history, cannot detect anomaly

        # Flatten update
        update_flat = self._flatten(update)

        # Compute mean of history
        history_flat = [self._flatten(h) for h in client_history]
        mean_history = np.mean(history_flat, axis=0)

        # Compute std of history
        if len(client_history) > 1:
            std_history = np.std(history_flat, axis=0)
            # Avoid division by zero
            std_history = np.maximum(std_history, 1e-8)
        else:
            std_history = np.ones_like(mean_history)

        # Compute z-score
        z_scores = np.abs((update_flat - mean_history) / (std_history + 1e-8))

        # Aggregate z-scores (use median for robustness)
        median_z_score = np.median(z_scores)

        # Set threshold (default from config)
        threshold = self.config['temporal_threshold']

        # Convert to anomaly score
        anomaly_score = min(1.0, median_z_score / threshold)

        return float(anomaly_score)

    def compute_combined_anomaly(self,
                                  update: np.ndarray,
                                  median_update: np.ndarray,
                                  global_direction: np.ndarray,
                                  client_id: str,
                                  client_history: Optional[List[np.ndarray]] = None,
                                  layer_names: Optional[List[str]] = None,
                                  factor_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute all factors and combine into single anomaly score.

        Args:
            update: Client's model update
            median_update: Median update across clients
            global_direction: Global update direction
            client_id: Client identifier
            client_history: Client's past updates
            layer_names: Optional layer names
            factor_weights: Weights for each factor

        Returns:
            Dictionary with individual factor scores and combined score
        """
        # Default weights
        if factor_weights is None:
            factor_weights = {
                'l2_magnitude': 0.30,
                'directional_consistency': 0.25,
                'layer_wise': 0.25,
                'temporal_consistency': 0.20
            }

        # Normalize weights
        total_weight = sum(factor_weights.values())
        factor_weights = {k: v / total_weight for k, v in factor_weights.items()}

        # Compute individual factors
        l2_score = self.compute_l2_anomaly(update, median_update)
        directional_score = self.compute_directional_anomaly(update, global_direction)

        layer_scores = self.compute_layer_anomaly(update, median_update, layer_names)
        # Average layer score
        layer_score = float(np.mean(list(layer_scores.values())))

        # Temporal consistency
        if client_history is not None and len(client_history) > 0:
            temporal_score = self.compute_temporal_anomaly(update, client_history)
        else:
            temporal_score = 0.0  # No history

        # Combined score (weighted sum)
        combined_score = (
            factor_weights['l2_magnitude'] * l2_score +
            factor_weights['directional_consistency'] * directional_score +
            factor_weights['layer_wise'] * layer_score +
            factor_weights['temporal_consistency'] * temporal_score
        )

        return {
            'l2_magnitude': l2_score,
            'directional_consistency': directional_score,
            'layer_wise': layer_score,
            'temporal_consistency': temporal_score,
            'combined': combined_score
        }

    @staticmethod
    def _flatten(update: np.ndarray) -> np.ndarray:
        """Flatten update to 1D array."""
        if isinstance(update, list):
            return np.concatenate([arr.flatten() for arr in update])
        return update.flatten()


class AnomalyThreshold:
    """
    Adaptive threshold computation for anomaly detection.

    Methods:
    - Percentile-based on training data
    - Fixed threshold
    - Online adaptive threshold
    """

    def __init__(self, method: str = 'percentile', percentile: float = 95):
        """
        Initialize AnomalyThreshold.

        Args:
            method: Threshold method ('percentile', 'fixed', 'adaptive')
            percentile: Percentile for percentile method
        """
        self.method = method
        self.percentile = percentile
        self.history: List[float] = []

    def update(self, scores: List[float]) -> None:
        """Update threshold with new scores."""
        self.history.extend(scores)

    def get_threshold(self) -> float:
        """Get current threshold."""
        if len(self.history) == 0:
            return 0.5  # Default

        if self.method == 'percentile':
            return np.percentile(self.history, self.percentile)
        elif self.method == 'fixed':
            return 0.5  # Default fixed
        elif self.method == 'adaptive':
            # Use running mean + 2*std
            return np.mean(self.history) + 2 * np.std(self.history)
        else:
            raise ValueError(f"Unknown threshold method: {self.method}")

    def is_anomalous(self, score: float) -> bool:
        """Check if score is above threshold."""
        return score > self.get_threshold()

    def reset(self) -> None:
        """Reset threshold history."""
        self.history.clear()
