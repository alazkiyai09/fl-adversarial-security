"""
Spectral (PCA-based) anomaly detector.
Detects outliers in reduced-dimensional space using PCA.
"""

from typing import List
import numpy as np
from sklearn.decomposition import PCA

from .base_detector import BaseDetector


class SpectralDetector(BaseDetector):
    """
    Detects malicious clients using PCA on model updates.

    Assumption: Honest clients' updates lie in low-dimensional subspace.
    Malicious clients project far from this subspace (high reconstruction error
    or extreme values in principal components).

    Detection:
    1. Fit PCA on baseline (honest) updates
    2. Project new updates to PCA space
    3. Flag outliers using z-score in PCA space
    """

    def __init__(
        self,
        n_components: int = 5,
        threshold: float = 3.0
    ):
        """
        Initialize spectral detector.

        Args:
            n_components: Number of principal components to keep
                          (typically much smaller than original dimensionality)
            threshold: Z-score threshold for outlier detection in PCA space
        """
        super().__init__(threshold=threshold)
        self.n_components = n_components

        # PCA model
        self.pca: PCA = None

        # Statistics in PCA space
        self.pca_mean: np.ndarray = None
        self.pca_std: np.ndarray = None

    def fit(self, baseline_updates: List[np.ndarray]) -> None:
        """
        Fit PCA model to honest client updates.

        Args:
            baseline_updates: List of flattened updates from honest clients
        """
        super().fit(baseline_updates)

        # Stack updates into matrix
        X = np.vstack(baseline_updates)

        # Fit PCA
        # Limit n_components to min(n_samples, n_features)
        actual_components = min(self.n_components, X.shape[0], X.shape[1])

        self.pca = PCA(n_components=actual_components, random_state=42)
        X_pca = self.pca.fit_transform(X)

        # Compute statistics in PCA space
        self.pca_mean = np.mean(X_pca, axis=0)
        self.pca_std = np.std(X_pca, axis=0)

        # Avoid division by zero
        self.pca_std[self.pca_std == 0] = 1e-8

    def compute_anomaly_score(self, update: np.ndarray, **kwargs) -> float:
        """
        Compute anomaly score based on PCA projection.

        Score = maximum z-score across all principal components

        Args:
            update: Flattened model update

        Returns:
            Anomaly score (higher = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before computing scores")

        # Project to PCA space
        update_pca = self.pca.transform(update.reshape(1, -1))[0]

        # Compute z-scores in PCA space
        z_scores = np.abs((update_pca - self.pca_mean) / self.pca_std)

        # Score = maximum z-score (any component being extreme is anomalous)
        score = float(np.max(z_scores))

        return score

    def get_reconstruction_error(self, update: np.ndarray) -> float:
        """
        Compute reconstruction error (distance from PCA subspace).

        High reconstruction error indicates update doesn't lie in
        the low-dimensional subspace of honest updates.

        Args:
            update: Flattened model update

        Returns:
            Reconstruction error (MSE between original and reconstructed)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before computing errors")

        # Project and reconstruct
        update_pca = self.pca.transform(update.reshape(1, -1))
        update_reconstructed = self.pca.inverse_transform(update_pca)[0]

        # Mean squared error
        error = float(np.mean((update - update_reconstructed) ** 2))

        return error

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get explained variance ratio for each principal component.

        Useful for understanding dimensionality of honest updates.

        Returns:
            Array of explained variance ratios
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted first")

        return self.pca.explained_variance_ratio_

    def get_pca_projection(self, update: np.ndarray) -> np.ndarray:
        """
        Get PCA projection of update (for visualization/analysis).

        Args:
            update: Flattened model update

        Returns:
            Projection in PCA space (n_components dimensional)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted first")

        return self.pca.transform(update.reshape(1, -1))[0]
