"""
Similarity computation utilities for FoolsGold defense.
"""

import numpy as np
from typing import List, Dict
from flwr.common import Parameters, parameters_to_ndarrays


def flatten_parameters(parameters: Parameters) -> np.ndarray:
    """
    Flatten Flower Parameters to 1D numpy array.

    Args:
        parameters: Flower Parameters object

    Returns:
        Flattened 1D numpy array of all parameters
    """
    # Convert Flower Parameters to list of numpy arrays
    ndarrays = parameters_to_ndarrays(parameters)

    # Concatenate all layers into single 1D vector
    flattened = np.concatenate([arr.flatten() for arr in ndarrays])

    return flattened


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    cos_sim = (v1 . v2) / (||v1|| * ||v2||)

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Handle edge case: zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)

    # Clip to handle numerical errors
    return float(np.clip(similarity, -1.0, 1.0))


def compute_pairwise_cosine_similarity(
    gradients: List[np.ndarray]
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix between gradients.

    Args:
        gradients: List of gradient vectors (1D numpy arrays)

    Returns:
        Similarity matrix S where S[i,j] = cos_sim(grad_i, grad_j)
        Shape: (num_clients, num_clients)
    """
    num_clients = len(gradients)

    # Handle edge case: single client
    if num_clients == 0:
        return np.array([])
    elif num_clients == 1:
        return np.array([[1.0]])

    # Initialize similarity matrix
    similarity_matrix = np.zeros((num_clients, num_clients))

    # Compute pairwise similarities
    for i in range(num_clients):
        for j in range(i, num_clients):
            sim = cosine_similarity(gradients[i], gradients[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric

    return similarity_matrix


def compute_similarity_from_history(
    history: Dict[int, List[np.ndarray]],
    client_ids: List[int]
) -> np.ndarray:
    """
    Compute pairwise similarity using historical gradients.

    For each pair of clients, computes similarity over their
    historical gradients (averaged across history).

    Args:
        history: Dictionary mapping client_id -> list of historical gradients
        client_ids: List of client IDs for this round

    Returns:
        Similarity matrix for current clients
    """
    num_clients = len(client_ids)

    if num_clients == 0:
        return np.array([])
    elif num_clients == 1:
        return np.array([[1.0]])

    # Get average gradient for each client from history
    avg_gradients = []
    for cid in client_ids:
        if cid in history and len(history[cid]) > 0:
            # Average historical gradients
            grad_list = np.array(history[cid])
            avg_grad = np.mean(grad_list, axis=0)
        else:
            # No history: use zero vector
            avg_grad = np.zeros_like(next(iter(history.values()))[0])
        avg_gradients.append(avg_grad)

    # Compute similarity on averaged gradients
    return compute_pairwise_cosine_similarity(avg_gradients)


def compute_adaptive_weights(
    similarity_matrix: np.ndarray,
    lr_scale_factor: float = 0.1
) -> np.ndarray:
    """
    Compute adaptive aggregation weights based on similarity.

    Clients with high similarity to others (potential Sybils) get
    reduced weights through learning rate adjustment.

    Args:
        similarity_matrix: Pairwise similarity matrix
        lr_scale_factor: Factor to scale learning rate

    Returns:
        Weight vector for aggregation
    """
    num_clients = similarity_matrix.shape[0]

    if num_clients == 0:
        return np.array([])
    elif num_clients == 1:
        return np.array([1.0])

    # Compute average similarity for each client
    avg_similarities = np.mean(similarity_matrix, axis=1)

    # Convert similarity to learning rate adjustment
    # Higher similarity -> lower contribution (higher denominator)
    # Formula: lr_k = lr / (1 + alpha * avg_similarity)
    weights = 1.0 / (1.0 + lr_scale_factor * avg_similarities)

    # Normalize weights to sum to num_clients (preserve scale)
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights * num_clients / weight_sum

    return weights
