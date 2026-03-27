"""
Utilities for parsing and preprocessing FL model updates.
Extracts PyTorch model parameters and converts to numpy arrays.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from flwr.common import Parameters

# Type alias for layer-wise updates
LayerUpdate = Dict[str, np.ndarray]


def extract_updates(parameters: Parameters) -> LayerUpdate:
    """
    Extract layer-wise updates from Flower Parameters object.

    Args:
        parameters: Flower Parameters object containing model updates

    Returns:
        Dictionary mapping layer names to numpy arrays of parameters

    Raises:
        ValueError: If parameters are malformed
    """
    if not parameters.tensors:
        raise ValueError("Parameters object is empty")

    # Deserialize parameters
    tensors = parameters.tensors
    layers = {}

    # Assume tensors are ordered consistently with model architecture
    # In practice, you'd need layer names from model state dict
    for i, tensor in enumerate(tensors):
        # Convert bytes to numpy array
        arr = np.frombuffer(tensor, dtype=np.float32)
        layers[f"layer_{i}"] = arr

    return layers


def flatten_update(update: LayerUpdate) -> np.ndarray:
    """
    Flatten layer-wise update into single 1D array.

    Args:
        update: Dictionary of layer_name -> parameter array

    Returns:
        Flattened 1D numpy array containing all parameters
    """
    if not update:
        raise ValueError("Update dictionary is empty")

    # Concatenate all layers in order
    flattened = np.concatenate([update[layer] for layer in sorted(update.keys())])
    return flattened


def unflatten_update(
    flattened: np.ndarray,
    reference_update: LayerUpdate
) -> LayerUpdate:
    """
    Reshape flattened update back to layer-wise structure.

    Args:
        flattened: Flattened 1D array of parameters
        reference_update: Reference update with layer structure

    Returns:
        Dictionary mapping layer names to reshaped arrays
    """
    layers = {}
    offset = 0

    for layer_name in sorted(reference_update.keys()):
        layer_size = reference_update[layer_name].size
        layers[layer_name] = flattened[offset:offset + layer_size]
        offset += layer_size

    if offset != flattened.size:
        raise ValueError(
            f"Size mismatch: reference has {offset} params, "
            f"flattened has {flattened.size}"
        )

    return layers


def extract_updates_from_pytorch(state_dict: Dict[str, torch.Tensor]) -> LayerUpdate:
    """
    Extract updates from PyTorch model state dict.

    Args:
        state_dict: PyTorch model state_dict

    Returns:
        Dictionary mapping layer names to numpy arrays
    """
    layers = {}
    for name, tensor in state_dict.items():
        layers[name] = tensor.detach().cpu().numpy().flatten()
    return layers


def compute_update_difference(
    old_params: LayerUpdate,
    new_params: LayerUpdate
) -> LayerUpdate:
    """
    Compute parameter update (difference) between two model states.

    Args:
        old_params: Previous model parameters
        new_params: New model parameters

    Returns:
        Layer-wise parameter differences (new - old)
    """
    if old_params.keys() != new_params.keys():
        raise ValueError("Parameter keys don't match")

    update = {}
    for layer_name in old_params.keys():
        update[layer_name] = new_params[layer_name] - old_params[layer_name]

    return update


def batch_flatten_updates(updates: List[LayerUpdate]) -> np.ndarray:
    """
    Flatten multiple updates into 2D array (num_clients x num_params).

    Useful for PCA, clustering, etc.

    Args:
        updates: List of layer-wise updates

    Returns:
        2D numpy array where each row is a flattened client update
    """
    return np.vstack([flatten_update(update) for update in updates])


def get_layer_sizes(update: LayerUpdate) -> Dict[str, int]:
    """
    Get size of each layer in the update.

    Args:
        update: Layer-wise update dictionary

    Returns:
        Dictionary mapping layer names to sizes
    """
    return {name: arr.size for name, arr in update.items()}


def total_parameters(update: LayerUpdate) -> int:
    """
    Count total number of parameters in update.

    Args:
        update: Layer-wise update dictionary

    Returns:
        Total number of parameters
    """
    return sum(arr.size for arr in update.values())
