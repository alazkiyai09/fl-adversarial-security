"""
Normalization utilities for FL model updates.
Ensures fair comparison across layers of different sizes.
"""

from typing import Dict, List
import numpy as np


LayerUpdate = Dict[str, np.ndarray]


def normalize_by_layer_size(update: LayerUpdate) -> LayerUpdate:
    """
    Normalize each layer by its size (number of parameters).

    This prevents large layers from dominating anomaly scores.

    Args:
        update: Layer-wise parameter update dictionary

    Returns:
        Normalized update with each layer divided by sqrt(num_params)
    """
    normalized = {}
    for layer_name, params in update.items():
        # Divide by sqrt(n) for L2 norm normalization
        scale = np.sqrt(params.size)
        normalized[layer_name] = params / scale

    return normalized


def normalize_by_frobenius_norm(update: LayerUpdate) -> LayerUpdate:
    """
    Normalize each layer by its Frobenius norm.

    Args:
        update: Layer-wise parameter update dictionary

    Returns:
        Normalized update with unit Frobenius norm per layer
    """
    normalized = {}
    for layer_name, params in update.items():
        norm = np.linalg.norm(params)
        if norm > 0:
            normalized[layer_name] = params / norm
        else:
            normalized[layer_name] = params.copy()

    return normalized


def normalize_by_std_baseline(
    update: LayerUpdate,
    baseline_stds: Dict[str, float]
) -> LayerUpdate:
    """
    Normalize update by standard deviations learned from baseline.

    Args:
        update: Layer-wise parameter update
        baseline_stds: Dictionary of layer_name -> std from honest clients

    Returns:
        Normalized update (z-score style per layer)
    """
    normalized = {}
    for layer_name, params in update.items():
        if layer_name in baseline_stds and baseline_stds[layer_name] > 0:
            normalized[layer_name] = params / baseline_stds[layer_name]
        else:
            normalized[layer_name] = params.copy()

    return normalized


def normalize_global(update: LayerUpdate) -> LayerUpdate:
    """
    Normalize entire update globally by its L2 norm.

    Args:
        update: Layer-wise parameter update

    Returns:
        Normalized update with unit L2 norm
    """
    flattened = np.concatenate([update[l] for l in sorted(update.keys())])
    norm = np.linalg.norm(flattened)

    if norm > 0:
        scale = 1.0 / norm
        return {name: arr * scale for name, arr in update.items()}
    else:
        return {name: arr.copy() for name, arr in update.items()}


def compute_layer_statistics(
    baseline_updates: List[LayerUpdate]
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std statistics per layer from baseline updates.

    Args:
        baseline_updates: List of layer-wise updates from honest clients

    Returns:
        Dictionary with layer_name -> {'mean': float, 'std': float}
    """
    if not baseline_updates:
        raise ValueError("baseline_updates cannot be empty")

    # Collect all layer values
    layer_data = {}
    for update in baseline_updates:
        for layer_name, params in update.items():
            if layer_name not in layer_data:
                layer_data[layer_name] = []
            layer_data[layer_name].append(params)

    # Compute statistics
    stats = {}
    for layer_name, param_list in layer_data.items():
        stacked = np.vstack(param_list)

        # Mean and std per parameter across clients
        stats[layer_name] = {
            'mean': np.mean(stacked, axis=0),
            'std': np.std(stacked, axis=0),
            'avg_norm': np.mean(np.linalg.norm(stacked, axis=1)),
            'std_norm': np.std(np.linalg.norm(stacked, axis=1))
        }

    return stats


def clip_by_norm(update: LayerUpdate, max_norm: float) -> LayerUpdate:
    """
    Clip update to have maximum L2 norm.

    Args:
        update: Layer-wise parameter update
        max_norm: Maximum allowed L2 norm

    Returns:
        Clipped update
    """
    flattened = np.concatenate([update[l] for l in sorted(update.keys())])
    norm = np.linalg.norm(flattened)

    if norm > max_norm:
        scale = max_norm / norm
        return {name: arr * scale for name, arr in update.items()}
    else:
        return {name: arr.copy() for name, arr in update.items()}
