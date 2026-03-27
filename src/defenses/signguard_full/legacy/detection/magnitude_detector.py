"""L2 norm magnitude anomaly detector."""

import torch
from typing import List
from scipy import stats

from src.defenses.signguard_full.legacy.detection.base import AnomalyDetector
from src.defenses.signguard_full.legacy.core.types import ModelUpdate
from src.defenses.signguard_full.legacy.utils.serialization import parameters_to_vector


class L2NormDetector(AnomalyDetector):
    """L2 norm-based anomaly detector.

    Detects anomalies by comparing the L2 norm (Euclidean magnitude) of
    parameter updates against expected values. Large magnitudes may indicate
    model poisoning attacks.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        use_mad: bool = True,
        mad_multiplier: float = 3.0,
        window_size: int = 10,
    ):
        """Initialize L2 norm detector.

        Args:
            threshold: Fixed anomaly threshold (if not using MAD)
            use_mad: Whether to use Median Absolute Deviation for adaptive threshold
            mad_multiplier: Number of MADs for threshold (z-score equivalent)
            window_size: Window size for computing statistics
        """
        self.threshold = threshold
        self.use_mad = use_mad
        self.mad_multiplier = mad_multiplier
        self.window_size = window_size
        self.norm_history: List[float] = []

    def compute_score(
        self,
        update: ModelUpdate,
        global_model: dict[str, torch.Tensor],
        client_history: List[ModelUpdate] | None = None,
    ) -> float:
        """Compute L2 norm-based anomaly score.

        Args:
            update: Client's model update
            global_model: Current global model
            client_history: Optional historical updates from this client

        Returns:
            Anomaly score in [0, 1], higher = more anomalous
        """
        # Compute update vector (difference from global model)
        update_diff = self._compute_update_diff(update.parameters, global_model)
        update_vector = parameters_to_vector(update_diff)

        # Compute L2 norm
        l2_norm = torch.norm(update_vector).item()

        # Normalize to [0, 1] based on statistics
        anomaly_score = self._normalize_norm(l2_norm)

        return anomaly_score

    def update_statistics(self, updates: List[ModelUpdate], global_model: dict[str, torch.Tensor]) -> None:
        """Update detector's internal statistics.

        Args:
            updates: List of model updates
            global_model: Current global model
        """
        norms = []
        for update in updates:
            update_diff = self._compute_update_diff(update.parameters, global_model)
            update_vector = parameters_to_vector(update_diff)
            norm = torch.norm(update_vector).item()
            norms.append(norm)

        # Update history
        self.norm_history.extend(norms)
        if len(self.norm_history) > self.window_size:
            self.norm_history = self.norm_history[-self.window_size:]

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

    def _normalize_norm(self, norm: float) -> float:
        """Normalize L2 norm to [0, 1] anomaly score.

        Args:
            norm: L2 norm value

        Returns:
            Normalized anomaly score
        """
        if self.use_mad and len(self.norm_history) >= 3:
            # Use Median Absolute Deviation
            median = torch.tensor(self.norm_history).median().item()
            mad = torch.tensor(
                [abs(n - median) for n in self.norm_history]
            ).median().item()
            
            # MAD to standard deviation approximation: std = 1.4826 * MAD
            # Avoid division by zero
            if mad < 1e-10:
                return 0.0
            
            # Compute z-score
            z_score = abs(norm - median) / (mad * 1.4826)
            
            # Convert to [0, 1] using sigmoid-like scaling
            # P-value approximation: 2 * (1 - Φ(z))
            # We use a simpler sigmoid for smoothness
            score = 2.0 * (1.0 / (1.0 + torch.exp(torch.tensor(-z_score / self.mad_multiplier))).item() - 0.5)
            return min(score, 1.0)
        
        elif len(self.norm_history) > 0:
            # Use mean and std
            mean = sum(self.norm_history) / len(self.norm_history)
            variance = sum((n - mean) ** 2 for n in self.norm_history) / len(self.norm_history)
            std = (variance ** 0.5)
            
            if std < 1e-10:
                return 0.0
            
            # Z-score based normalization
            z_score = abs(norm - mean) / std
            score = 1.0 - torch.exp(torch.tensor(-z_score ** 2 / 10)).item()
            return min(score, 1.0)
        
        else:
            # No history, use simple scaling
            # Assume norms are typically in [0, 10]
            return min(norm / 10.0, 1.0)

    def get_statistics(self) -> dict:
        """Get current detector statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.norm_history:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        import numpy as np
        norms = np.array(self.norm_history)
        return {
            "count": len(norms),
            "mean": float(np.mean(norms)),
            "median": float(np.median(norms)),
            "std": float(np.std(norms)),
            "min": float(np.min(norms)),
            "max": float(np.max(norms)),
        }
