"""
Layer-wise anomaly detector.
Analyzes each layer independently to identify specific anomalous layers.
"""

from typing import List, Dict
import numpy as np

from .base_detector import BaseDetector


LayerUpdate = Dict[str, np.ndarray]


class LayerwiseDetector(BaseDetector):
    """
    Detects malicious clients by analyzing each layer independently.

    Assumption: Malicious updates often target specific layers
    (e.g., last layer for backdoor attacks).

    Detection:
    1. Compute L2 norm per layer
    2. Flag layers with z-score > threshold
    3. Client flagged if >= min_anomalous_layers are anomalous
    """

    def __init__(
        self,
        layer_threshold: float = 3.0,
        min_anomalous_layers: int = 2
    ):
        """
        Initialize layer-wise detector.

        Args:
            layer_threshold: Z-score threshold for individual layers
            min_anomalous_layers: Minimum number of anomalous layers to flag client
        """
        # Store both thresholds
        super().__init__(threshold=layer_threshold)
        self.min_anomalous_layers = min_anomalous_layers

        # Learned statistics per layer
        self.layer_means: Dict[str, float] = {}
        self.layer_stds: Dict[str, float] = {}

    def fit(self, baseline_updates: List[LayerUpdate]) -> None:
        """
        Learn layer-wise norm statistics from honest clients.

        Args:
            baseline_updates: List of layer-wise updates from honest clients
        """
        # Validate
        if not baseline_updates:
            raise ValueError("baseline_updates cannot be empty")

        # Collect norms per layer
        layer_norms = {}
        for update in baseline_updates:
            for layer_name, params in update.items():
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append(np.linalg.norm(params))

        # Compute statistics per layer
        for layer_name, norms in layer_norms.items():
            self.layer_means[layer_name] = float(np.mean(norms))
            self.layer_stds[layer_name] = float(np.std(norms))
            if self.layer_stds[layer_name] == 0:
                self.layer_stds[layer_name] = 1e-8  # Avoid division by zero

        self.is_fitted = True

    def compute_anomaly_score(self, update: LayerUpdate, **kwargs) -> float:
        """
        Compute anomaly score based on layer-wise analysis.

        Score = average z-score of all anomalous layers
        (layers with z-score > threshold)

        Args:
            update: Layer-wise parameter update

        Returns:
            Anomaly score (higher = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before computing scores")

        # Compute z-scores per layer
        layer_z_scores = self._compute_layer_z_scores(update)

        # Anomalous layers
        anomalous_layers = {
            layer: z_score
            for layer, z_score in layer_z_scores.items()
            if abs(z_score) > self.threshold
        }

        # Score: average z-score of anomalous layers
        if anomalous_layers:
            score = np.mean(list(anomalous_layers.values()))
        else:
            score = 0.0

        return float(score)

    def is_malicious(self, update: LayerUpdate, **kwargs) -> bool:
        """
        Binary decision: is client malicious?

        Client flagged if >= min_anomalous_layers are anomalous.

        Args:
            update: Layer-wise parameter update

        Returns:
            True if client is flagged as malicious
        """
        layer_z_scores = self._compute_layer_z_scores(update)

        # Count anomalous layers
        num_anomalous = sum(
            1 for z_score in layer_z_scores.values()
            if abs(z_score) > self.threshold
        )

        return num_anomalous >= self.min_anomalous_layers

    def get_anomalous_layers(self, update: LayerUpdate) -> List[str]:
        """
        Get list of anomalous layer names.

        Args:
            update: Layer-wise parameter update

        Returns:
            List of layer names with z-score > threshold
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before analysis")

        layer_z_scores = self._compute_layer_z_scores(update)

        anomalous = [
            layer_name
            for layer_name, z_score in layer_z_scores.items()
            if abs(z_score) > self.threshold
        ]

        return anomalous

    def _compute_layer_z_scores(self, update: LayerUpdate) -> Dict[str, float]:
        """
        Compute z-score for each layer's L2 norm.

        Args:
            update: Layer-wise parameter update

        Returns:
            Dictionary mapping layer_name -> z_score
        """
        z_scores = {}

        for layer_name, params in update.items():
            if layer_name not in self.layer_means:
                # New layer not seen in baseline (shouldn't happen)
                continue

            norm = np.linalg.norm(params)
            mean = self.layer_means[layer_name]
            std = self.layer_stds[layer_name]

            z_scores[layer_name] = (norm - mean) / std

        return z_scores

    def get_layer_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get learned layer statistics (for analysis).

        Returns:
            Dictionary with layer_name -> {'mean': float, 'std': float}
        """
        return {
            layer: {'mean': self.layer_means[layer], 'std': self.layer_stds[layer]}
            for layer in self.layer_means
        }
