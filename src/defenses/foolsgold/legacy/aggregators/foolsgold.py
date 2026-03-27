"""
FoolsGold: Sybil-Resistant Federated Learning Aggregation

Paper: "Mitigating Sybils in Federated Learning Poisoning"
       Fung et al., 2020 (AISTATS)

Key insight: Sybil attackers send similar updates (coordinated).
FoolsGold reduces contribution weight of clients with similar gradients.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from flwr.common import Parameters, FitRes, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from .base import BaseAggregator
from ..utils.similarity import (
    flatten_parameters,
    compute_pairwise_cosine_similarity,
    compute_similarity_from_history
)


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

    # Handle edge case: no clients
    if num_clients == 0:
        return np.array([])

    # Handle edge case: single client
    if num_clients == 1:
        return np.array([[1.0]])

    # Initialize similarity matrix
    similarity_matrix = np.zeros((num_clients, num_clients))

    # Compute pairwise similarities
    for i in range(num_clients):
        for j in range(i, num_clients):
            # Cosine similarity
            norm_i = np.linalg.norm(gradients[i])
            norm_j = np.linalg.norm(gradients[j])

            # Handle zero vectors
            if norm_i == 0 or norm_j == 0:
                sim = 0.0
            else:
                dot_product = np.dot(gradients[i], gradients[j])
                sim = dot_product / (norm_i * norm_j)
                sim = float(np.clip(sim, -1.0, 1.0))

            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    return similarity_matrix


def compute_contribution_scores(
    similarity_matrix: np.ndarray,
    history: Dict[int, List[np.ndarray]],
    history_length: int,
    client_ids: List[int]
) -> np.ndarray:
    """
    Compute FoolsGold contribution scores (alpha_k) for each client.

    Algorithm:
    1. Retrieve historical gradients for each client (up to history_length)
    2. Compute pairwise cosine similarity using historical gradients
    3. Convert similarity to contribution scores
       - High similarity -> Low contribution (potential Sybil)
       - Low similarity -> High contribution (honest/unique)

    Args:
        similarity_matrix: Current round similarity matrix
        history: Historical gradients dict {client_id: [grad_1, grad_2, ...]}
        history_length: Maximum number of historical gradients to use
        client_ids: List of client IDs for current round

    Returns:
        Contribution scores alpha_k for each client
        Shape: (num_clients,)
    """
    num_clients = len(client_ids)

    # Edge case: no clients
    if num_clients == 0:
        return np.array([])

    # Edge case: single client
    if num_clients == 1:
        return np.array([1.0])

    # Build historical gradient lists for each client
    historical_grads = []
    for cid in client_ids:
        if cid in history and len(history[cid]) > 0:
            # Use up to history_length most recent gradients
            grads = history[cid][-history_length:]
            # Average historical gradients
            avg_grad = np.mean(grads, axis=0)
            historical_grads.append(avg_grad)
        else:
            # No history - use zero vector (will have low similarity)
            if len(historical_grads) > 0:
                historical_grads.append(np.zeros_like(historical_grads[0]))
            else:
                # First round - use current similarity
                historical_grads.append(np.zeros(10))  # Placeholder

    # Compute similarity on historical gradients
    hist_similarity = compute_pairwise_cosine_similarity(historical_grads)

    # Compute contribution scores
    # Formula: alpha_k = 1 / (1 + sum_j S[k,j] * beta_k)
    # where beta_k scales with similarity
    alpha_scores = np.zeros(num_clients)

    for k in range(num_clients):
        # Sum of similarities to other clients
        sim_sum = np.sum(hist_similarity[k, :])

        # FoolsGold formula: higher similarity -> lower alpha
        # alpha_k represents contribution weight
        alpha_k = 1.0 / (1.0 + sim_sum)
        alpha_scores[k] = alpha_k

    # Normalize to preserve total weight
    alpha_sum = np.sum(alpha_scores)
    if alpha_sum > 0:
        alpha_scores = alpha_scores * num_clients / alpha_sum

    return alpha_scores


def foolsgold_aggregate(
    parameters: List[Parameters],
    contribution_scores: np.ndarray,
    lr_scale_factor: float = 1.0
) -> Parameters:
    """
    Weighted aggregation with learning rate adjustment.

    Formula:
        w_global = sum_k (alpha_k * lr_k) * w_k

    where lr_k = lr / (1 + alpha_k) to further reduce high-similarity clients

    Args:
        parameters: List of client parameters to aggregate
        contribution_scores: Alpha scores for each client
        lr_scale_factor: Learning rate scaling factor

    Returns:
        Aggregated global parameters
    """
    num_clients = len(parameters)

    if num_clients == 0:
        raise ValueError("Cannot aggregate empty parameter list")

    # Convert all parameters to numpy arrays
    all_ndarrays = []
    for params in parameters:
        ndarrays = [arr.copy() for arr in params.tensors]
        all_ndarrays.append(ndarrays)

    # Get layer structure from first client
    num_layers = len(all_ndarrays[0])

    # Aggregate each layer
    aggregated_ndarrays = []
    for layer_idx in range(num_layers):
        layer_params = [all_ndarrays[c][layer_idx] for c in range(num_clients)]

        # Compute weighted average with contribution scores
        weighted_sum = np.zeros_like(layer_params[0])
        total_weight = 0.0

        for client_idx, params in enumerate(layer_params):
            # Contribution score
            alpha = contribution_scores[client_idx]

            # Learning rate adjustment
            # Higher alpha (more unique) -> higher effective lr
            lr_adjustment = 1.0 / (1.0 + lr_scale_factor * (1.0 - alpha))

            weight = alpha * lr_adjustment
            weighted_sum += weight * params
            total_weight += weight

        # Normalize
        if total_weight > 0:
            aggregated_layer = weighted_sum / total_weight
        else:
            # Fallback to simple average
            aggregated_layer = np.mean(layer_params, axis=0)

        aggregated_ndarrays.append(aggregated_layer)

    return ndarrays_to_parameters(aggregated_ndarrays)


class FoolsGoldAggregator(BaseAggregator):
    """
    FoolsGold aggregator for Sybil-resistant federated learning.

    Maintains history of client gradients and computes adaptive
    contribution scores based on pairwise cosine similarity.
    """

    def __init__(
        self,
        history_length: int = 10,
        similarity_threshold: float = 0.9,
        lr_scale_factor: float = 0.1
    ):
        """
        Initialize FoolsGold aggregator.

        Args:
            history_length: Maximum number of historical gradients to track
            similarity_threshold: Threshold for flagging high similarity
            lr_scale_factor: Factor for learning rate adjustment (0-1)
        """
        super().__init__(
            history_length=history_length,
            similarity_threshold=similarity_threshold,
            lr_scale_factor=lr_scale_factor
        )
        self.history_length = history_length
        self.similarity_threshold = similarity_threshold
        self.lr_scale_factor = lr_scale_factor

        # Gradient history: {client_id: [grad_1, grad_2, ...]}
        self.gradient_history: Dict[int, List[np.ndarray]] = defaultdict(list)

        # Track metrics
        self.history["similarity_matrices"] = []
        self.history["contribution_scores"] = []
        self.history["flagged_sybils"] = []

    def aggregate(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Parameters:
        """
        Aggregate client updates using FoolsGold.

        Args:
            results: List of (client_proxy, fit_result) tuples

        Returns:
            Aggregated global parameters
        """
        if not results:
            raise ValueError("No results to aggregate")

        # Extract client IDs and parameters
        client_ids = []
        parameters_list = []
        current_gradients = []

        for client_proxy, fit_res in results:
            cid = int(str(client_proxy).split("_")[-1])
            client_ids.append(cid)
            parameters_list.append(fit_res.parameters)

            # Flatten and store gradient
            grad = flatten_parameters(fit_res.parameters)
            current_gradients.append(grad)

            # Update history
            self.gradient_history[cid].append(grad)
            # Trim to history_length
            if len(self.gradient_history[cid]) > self.history_length:
                self.gradient_history[cid] = self.gradient_history[cid][-self.history_length:]

        # Compute pairwise similarity
        similarity_matrix = compute_pairwise_cosine_similarity(current_gradients)
        self.history["similarity_matrices"].append(similarity_matrix.copy())

        # Compute contribution scores
        contribution_scores = compute_contribution_scores(
            similarity_matrix,
            dict(self.gradient_history),
            self.history_length,
            client_ids
        )
        self.history["contribution_scores"].append(contribution_scores.copy())

        # Flag potential Sybils (high similarity)
        flagged = []
        for i in range(len(client_ids)):
            avg_sim = np.mean(similarity_matrix[i, :])
            if avg_sim > self.similarity_threshold:
                flagged.append(client_ids[i])
        self.history["flagged_sybils"].append(flagged)

        # Aggregate with contribution scores
        aggregated = foolsgold_aggregate(
            parameters_list,
            contribution_scores,
            self.lr_scale_factor
        )

        return aggregated

    def reset_history(self) -> None:
        """Reset gradient history."""
        self.gradient_history.clear()
        self.history["similarity_matrices"] = []
        self.history["contribution_scores"] = []
        self.history["flagged_sybils"] = []
