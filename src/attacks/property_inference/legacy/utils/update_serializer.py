"""
Update Serializer - Extract and serialize model updates.

This module provides utilities for converting between model updates
and numpy arrays for meta-classifier training.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Union
from ..fl_system.model import FraudDetectionModel


def extract_weights(model: FraudDetectionModel) -> np.ndarray:
    """Flatten model weights into 1D array.

    Args:
        model: PyTorch model

    Returns:
        1D numpy array of all weights

    Example:
        >>> weights = extract_weights(model)
        >>> weights.shape
        (120,)  # Total number of parameters
    """
    return model.get_weights()


def extract_gradients(model: FraudDetectionModel) -> np.ndarray:
    """Flatten model gradients into 1D array.

    Args:
        model: PyTorch model

    Returns:
        1D numpy array of all gradients
    """
    return model.get_gradients()


def serialize_update(update: Dict[str, Any]) -> np.ndarray:
    """Serialize model update dict to array.

    Args:
        update: Dict containing 'weights' or 'update' key

    Returns:
        1D numpy array

    Example:
        >>> update = {'weights': np.array([1, 2, 3]), 'n_samples': 100}
        >>> serialized = serialize_update(update)
        >>> serialized
        array([1., 2., 3.])
    """
    if 'weights' in update:
        return update['weights'].flatten()
    elif 'update' in update:
        return update['update'].flatten()
    else:
        raise ValueError("Update dict must contain 'weights' or 'update' key")


def deserialize_update(
    array: np.ndarray,
    template_model: FraudDetectionModel
) -> Dict[str, Any]:
    """Deserialize array back to model update dict.

    Args:
        array: 1D numpy array of weights
        template_model: Model to use as template for structure

    Returns:
        Dict with 'weights' key containing reshaped array

    Example:
        >>> weights = np.array([1, 2, 3, 4, 5])
        >>> update = deserialize_update(weights, model)
        >>> update['weights'].shape
        (5,)
    """
    return {'weights': array}


def serialize_batch_updates(
    updates: List[Dict[str, Any]]
) -> np.ndarray:
    """Serialize multiple updates into 2D array.

    Args:
        updates: List of update dicts

    Returns:
        2D numpy array of shape (n_updates, n_params)

    Example:
        >>> updates = [
        ...     {'weights': np.array([1, 2, 3])},
        ...     {'weights': np.array([4, 5, 6])}
        ... ]
        >>> batch = serialize_batch_updates(updates)
        >>> batch.shape
        (2, 3)
    """
    return np.array([serialize_update(u) for u in updates])


def normalize_updates(
    updates: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """Normalize model updates.

    Args:
        updates: 2D array of shape (n_samples, n_params)
        method: Normalization method ('standard', 'minmax', 'l2')

    Returns:
        Normalized updates

    Example:
        >>> updates = np.random.randn(100, 50)
        >>> normalized = normalize_updates(updates, method='standard')
        >>> normalized.mean(axis=0)
        array([~0, ~0, ...])  # Near zero
    """
    if method == 'standard':
        # Z-score normalization
        mean = updates.mean(axis=0)
        std = updates.std(axis=0)
        return (updates - mean) / (std + 1e-8)

    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = updates.min(axis=0)
        max_val = updates.max(axis=0)
        return (updates - min_val) / (max_val - min_val + 1e-8)

    elif method == 'l2':
        # L2 normalization per sample
        norms = np.linalg.norm(updates, axis=1, keepdims=True)
        return updates / (norms + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_update_statistics(
    updates: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute statistics across updates.

    Args:
        updates: 2D array of shape (n_samples, n_params)

    Returns:
        Dict with 'mean', 'std', 'min', 'max' arrays

    Example:
        >>> updates = np.random.randn(100, 50)
        >>> stats = compute_update_statistics(updates)
        >>> stats['mean'].shape
        (50,)
    """
    return {
        'mean': updates.mean(axis=0),
        'std': updates.std(axis=0),
        'min': updates.min(axis=0),
        'max': updates.max(axis=0)
    }


def reduce_dimensionality(
    updates: np.ndarray,
    method: str = 'pca',
    n_components: int = 50
) -> np.ndarray:
    """Reduce dimensionality of updates.

    Useful for meta-classifier training when updates are high-dimensional.

    Args:
        updates: 2D array of shape (n_samples, n_params)
        method: Reduction method ('pca', 'random_projection')
        n_components: Target dimensionality

    Returns:
        Reduced array of shape (n_samples, n_components)

    Example:
        >>> updates = np.random.randn(100, 1000)
        >>> reduced = reduce_dimensionality(updates, method='pca', n_components=50)
        >>> reduced.shape
        (100, 50)
    """
    if method == 'pca':
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        return pca.fit_transform(updates)

    elif method == 'random_projection':
        from sklearn.random_projection import GaussianRandomProjection

        transformer = GaussianRandomProjection(n_components=n_components)
        return transformer.fit_transform(updates)

    else:
        raise ValueError(f"Unknown reduction method: {method}")


def compute_update_similarity(
    update1: np.ndarray,
    update2: np.ndarray,
    metric: str = 'cosine'
) -> float:
    """Compute similarity between two updates.

    Args:
        update1: First update array
        update2: Second update array
        metric: Similarity metric ('cosine', 'euclidean', 'correlation')

    Returns:
        Similarity score
    """
    if metric == 'cosine':
        return np.dot(update1, update2) / (
            np.linalg.norm(update1) * np.linalg.norm(update2) + 1e-8
        )
    elif metric == 'euclidean':
        return -np.linalg.norm(update1 - update2)
    elif metric == 'correlation':
        return np.corrcoef(update1, update2)[0, 1]
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_update_distances(
    updates: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """Compute pairwise distances between updates.

    Args:
        updates: 2D array of shape (n_samples, n_params)
        metric: Distance metric ('euclidean', 'cosine', 'manhattan')

    Returns:
        Distance matrix of shape (n_samples, n_samples)

    Example:
        >>> updates = np.random.randn(10, 50)
        >>> distances = compute_update_distances(updates)
        >>> distances.shape
        (10, 10)
    """
    from scipy.spatial.distance import pdist, squareform

    condensed = pdist(updates, metric=metric)
    return squareform(condensed)


def extract_update_features(
    updates: np.ndarray,
    feature_types: List[str] = None
) -> np.ndarray:
    """Extract additional features from updates.

    Args:
        updates: 2D array of shape (n_samples, n_params)
        feature_types: Types of features to extract
            Options: ['mean', 'std', 'l2_norm', 'max', 'min', 'median']

    Returns:
        Feature array of shape (n_samples, n_features)

    Example:
        >>> updates = np.random.randn(100, 50)
        >>> features = extract_update_features(
        ...     updates,
        ...     feature_types=['mean', 'std', 'l2_norm']
        ... )
        >>> features.shape
        (100, 3)
    """
    if feature_types is None:
        feature_types = ['mean', 'std', 'l2_norm']

    features = []

    for update in updates:
        sample_features = []

        if 'mean' in feature_types:
            sample_features.append(update.mean())
        if 'std' in feature_types:
            sample_features.append(update.std())
        if 'l2_norm' in feature_types:
            sample_features.append(np.linalg.norm(update))
        if 'max' in feature_types:
            sample_features.append(update.max())
        if 'min' in feature_types:
            sample_features.append(update.min())
        if 'median' in feature_types:
            sample_features.append(np.median(update))

        features.append(sample_features)

    return np.array(features)


def save_updates(
    updates: np.ndarray,
    filepath: str
) -> None:
    """Save updates to file.

    Args:
        updates: Updates array
        filepath: Path to save file (.npy or .npz)
    """
    if filepath.endswith('.npy'):
        np.save(filepath, updates)
    elif filepath.endswith('.npz'):
        np.savez_compressed(filepath, updates=updates)
    else:
        raise ValueError("Filepath must end with .npy or .npz")


def load_updates(filepath: str) -> np.ndarray:
    """Load updates from file.

    Args:
        filepath: Path to load file

    Returns:
        Updates array
    """
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        return data['updates']
    else:
        raise ValueError("Filepath must end with .npy or .npz")
