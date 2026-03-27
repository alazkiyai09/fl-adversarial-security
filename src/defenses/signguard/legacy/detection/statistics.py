"""
Online Statistics for SignGuard Anomaly Detection

Maintains running statistics (mean, variance, etc.) for online anomaly detection.
Supports exponential moving average and incremental variance computation.
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np


class OnlineStatistics:
    """
    Maintains online statistics for anomaly detection.

    Features:
    - Exponential moving average (EMA)
    - Incremental mean and variance (Welford's algorithm)
    - Per-client and per-layer tracking
    - Temporal window for historical patterns
    """

    def __init__(self, ema_alpha: float = 0.1,
                 temporal_window: int = 5):
        """
        Initialize OnlineStatistics.

        Args:
            ema_alpha: EMA decay factor (lower = more smoothing)
            temporal_window: Number of past rounds to track
        """
        self.ema_alpha = ema_alpha
        self.temporal_window = temporal_window

        # Global statistics
        self.global_mean: Optional[np.ndarray] = None
        self.global_variance: Optional[np.ndarray] = None
        self.global_count: int = 0

        # EMA of updates
        self.global_ema: Optional[np.ndarray] = None

        # Per-client tracking
        self.client_history: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.client_emas: Dict[str, np.ndarray] = {}
        self.client_means: Dict[str, np.ndarray] = {}
        self.client_variances: Dict[str, np.ndarray] = {}

        # Layer-wise statistics
        self.layer_means: Dict[str, np.ndarray] = defaultdict(list)
        self.layer_emas: Dict[str, float] = {}

    def update_global(self, update: np.ndarray) -> None:
        """
        Update global statistics with a new update.

        Args:
            update: Model update (flattened or as array)
        """
        update_flat = self._flatten(update)

        # Initialize if first update
        if self.global_mean is None:
            self.global_mean = np.zeros_like(update_flat)
            self.global_variance = np.zeros_like(update_flat)
            self.global_ema = np.zeros_like(update_flat)

        # Update EMA
        self.global_ema = (self.ema_alpha * update_flat +
                          (1 - self.ema_alpha) * self.global_ema)

        # Update mean and variance (Welford's algorithm)
        self.global_count += 1
        delta = update_flat - self.global_mean
        self.global_mean += delta / self.global_count
        delta2 = update_flat - self.global_mean
        self.global_variance += delta * delta2

    def update_client(self, client_id: str, update: np.ndarray) -> None:
        """
        Update per-client statistics.

        Args:
            client_id: Client identifier
            update: Model update
        """
        update_flat = self._flatten(update)

        # Update history (maintain sliding window)
        self.client_history[client_id].append(update_flat)
        if len(self.client_history[client_id]) > self.temporal_window:
            self.client_history[client_id].pop(0)

        # Update EMA
        if client_id not in self.client_emas:
            self.client_emas[client_id] = np.zeros_like(update_flat)

        self.client_emas[client_id] = (
            self.ema_alpha * update_flat +
            (1 - self.ema_alpha) * self.client_emas[client_id]
        )

        # Update mean and variance
        history = self.client_history[client_id]
        self.client_means[client_id] = np.mean(history, axis=0)
        if len(history) > 1:
            self.client_variances[client_id] = np.var(history, axis=0)
        else:
            self.client_variances[client_id] = np.zeros_like(update_flat)

    def update_layer(self, layer_name: str, layer_update: np.ndarray) -> None:
        """
        Update layer-wise statistics.

        Args:
            layer_name: Name/identifier for the layer
            layer_update: Layer parameters
        """
        # Compute L2 norm of layer update
        layer_norm = np.linalg.norm(layer_update)

        # Update history
        self.layer_means[layer_name].append(layer_norm)
        if len(self.layer_means[layer_name]) > self.temporal_window:
            self.layer_means[layer_name].pop(0)

        # Update EMA
        if layer_name not in self.layer_emas:
            self.layer_emas[layer_name] = layer_norm
        else:
            self.layer_emas[layer_name] = (
                self.ema_alpha * layer_norm +
                (1 - self.ema_alpha) * self.layer_emas[layer_name]
            )

    def get_global_mean(self) -> Optional[np.ndarray]:
        """Get global mean update."""
        return self.global_mean

    def get_global_variance(self) -> Optional[np.ndarray]:
        """Get global variance (unnormalized)."""
        return self.global_variance

    def get_global_std(self) -> Optional[np.ndarray]:
        """Get global standard deviation."""
        if self.global_variance is None or self.global_count < 2:
            return None
        return np.sqrt(self.global_variance / (self.global_count - 1))

    def get_global_ema(self) -> Optional[np.ndarray]:
        """Get global EMA of updates."""
        return self.global_ema

    def get_client_ema(self, client_id: str) -> Optional[np.ndarray]:
        """Get EMA for a specific client."""
        return self.client_emas.get(client_id)

    def get_client_mean(self, client_id: str) -> Optional[np.ndarray]:
        """Get mean update for a specific client."""
        return self.client_means.get(client_id)

    def get_client_std(self, client_id: str) -> Optional[np.ndarray]:
        """Get std of updates for a specific client."""
        var = self.client_variances.get(client_id)
        return np.sqrt(var) if var is not None else None

    def get_client_history(self, client_id: str) -> List[np.ndarray]:
        """Get historical updates for a client."""
        return self.client_history.get(client_id, [])

    def get_layer_ema(self, layer_name: str) -> Optional[float]:
        """Get EMA for a specific layer."""
        return self.layer_emas.get(layer_name)

    def get_layer_history(self, layer_name: str) -> List[float]:
        """Get history of L2 norms for a layer."""
        return self.layer_means.get(layer_name, [])

    def compute_median_update(self, updates: List[np.ndarray]) -> np.ndarray:
        """
        Compute median update (robust to outliers).

        Args:
            updates: List of model updates

        Returns:
            Median update (element-wise median)
        """
        # Stack all updates and compute median
        stacked = np.stack([self._flatten(u) for u in updates])
        return np.median(stacked, axis=0)

    def compute_mad(self, values: np.ndarray) -> float:
        """
        Compute Median Absolute Deviation (robust std estimator).

        Args:
            values: Array of values

        Returns:
            MAD value
        """
        median = np.median(values)
        return np.median(np.abs(values - median))

    def reset_global(self) -> None:
        """Reset global statistics."""
        self.global_mean = None
        self.global_variance = None
        self.global_ema = None
        self.global_count = 0

    def reset_client(self, client_id: str) -> None:
        """Reset statistics for a specific client."""
        if client_id in self.client_history:
            del self.client_history[client_id]
        if client_id in self.client_emas:
            del self.client_emas[client_id]
        if client_id in self.client_means:
            del self.client_means[client_id]
        if client_id in self.client_variances:
            del self.client_variances[client_id]

    def reset_all(self) -> None:
        """Reset all statistics."""
        self.global_mean = None
        self.global_variance = None
        self.global_ema = None
        self.global_count = 0
        self.client_history.clear()
        self.client_emas.clear()
        self.client_means.clear()
        self.client_variances.clear()
        self.layer_means.clear()
        self.layer_emas.clear()

    @staticmethod
    def _flatten(update: np.ndarray) -> np.ndarray:
        """
        Flatten update to 1D array.

        Args:
            update: Model update (potentially multi-dimensional)

        Returns:
            Flattened 1D array
        """
        if isinstance(update, list):
            # Handle list of arrays
            return np.concatenate([arr.flatten() for arr in update])
        return update.flatten()


class RunningStats:
    """
    Simple running statistics (mean, variance, min, max).

    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self):
        """Initialize RunningStats."""
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0  # Sum of squared deviations
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def update(self, value: float) -> None:
        """
        Update with a new value.

        Args:
            value: New observation
        """
        self.count += 1

        # Update mean and M2 (Welford's algorithm)
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

        # Update min/max
        if self.min_val is None or value < self.min_val:
            self.min_val = value
        if self.max_val is None or value > self.max_val:
            self.max_val = value

    def get_mean(self) -> float:
        """Get current mean."""
        return self.mean

    def get_variance(self) -> float:
        """Get current variance (unbiased estimator)."""
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0

    def get_std(self) -> float:
        """Get current standard deviation."""
        return np.sqrt(self.get_variance())

    def get_min(self) -> Optional[float]:
        """Get minimum value."""
        return self.min_val

    def get_max(self) -> Optional[float]:
        """Get maximum value."""
        return self.max_val

    def get_count(self) -> int:
        """Get number of observations."""
        return self.count

    def reset(self) -> None:
        """Reset statistics."""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = None
        self.max_val = None
