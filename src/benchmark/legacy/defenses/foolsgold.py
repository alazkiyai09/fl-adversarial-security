"""
FoolsGold defense implementation.

FoolsGold identifies and down-weights malicious clients based on the similarity
of their updates across multiple rounds. Colluding attackers have highly similar
updates, which FoolsGold detects and penalizes.

Reference: Blanchard et al. "Machine learning with adversaries: Byzantine
tolerant gradient descent." NIPS 2017 (FoolsGold extension).
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from numpy.typing import NDArray

from .base import BaseDefense


class FoolsGoldDefense(BaseDefense):
    """
    FoolsGold aggregation with history-based similarity scoring.

    Maintains a history of client updates across rounds and uses the
    similarity of these updates to compute adaptive weights for each client.
    Malicious clients that collude will have similar updates and get
    lower weights.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FoolsGold defense.

        Args:
            config: Configuration with keys:
                - history_length: Number of rounds of history to keep (default: 10)
                - min_weight: Minimum weight for any client (default: 0.01)
        """
        super().__init__(config)
        self.history_length = config.get("history_length", 10)
        self.min_weight = config.get("min_weight", 0.01)

        # History storage: dict mapping client_id to list of past updates
        self._history: Dict[int, List[NDArray]] = {}

        # Current round detection metrics
        self._current_weights: Optional[NDArray] = None
        self._current_scores: Optional[NDArray] = None

    def reset_state(self) -> None:
        """Reset defense state between experiments."""
        self._history.clear()
        self._current_weights = None
        self._current_scores = None

    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate updates using FoolsGold weighted averaging.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters with FoolsGold weights
        """
        if len(updates) == 0:
            return np.array([])

        # Extract client IDs and parameters
        client_ids = [client_id for client_id, _ in updates]
        params = self._extract_updates(updates)
        n_clients = len(params)

        # Update history with current updates
        self._update_history(client_ids, params)

        # Compute FoolsGold weights
        weights = self._compute_weights(client_ids, params)

        # Apply weighted average
        aggregated = np.average(params, axis=0, weights=weights)

        # Store for detection metrics
        self._current_weights = weights

        return aggregated

    def _update_history(self, client_ids: List[int], params: NDArray) -> None:
        """
        Update history with current round's updates.

        Args:
            client_ids: List of client IDs
            params: Current round parameters
        """
        for client_id, param in zip(client_ids, params):
            if client_id not in self._history:
                self._history[client_id] = []

            self._history[client_id].append(param.copy())

            # Trim history if too long
            if len(self._history[client_id]) > self.history_length:
                self._history[client_id] = self._history[client_id][-self.history_length:]

    def _compute_weights(self, client_ids: List[int], params: NDArray) -> NDArray:
        """
        Compute FoolsGold weights based on update similarity.

        Args:
            client_ids: List of client IDs
            params: Current round parameters

        Returns:
            Weight array of shape (n_clients,)
        """
        n_clients = len(params)

        # If insufficient history, use uniform weights
        if len(self._history) < 2:
            return np.ones(n_clients) / n_clients

        # Compute similarity scores using history
        similarity_matrix = self._compute_similarity_matrix(client_ids)

        # Compute FoolsGold scores
        # For each client i, score_i = sum_j(cos_similarity_ij)
        # Higher total similarity means more colluding (likely malicious)
        scores = np.sum(similarity_matrix, axis=1)

        # Convert scores to weights (inverse relationship)
        # Higher similarity -> lower weight
        weights = self._scores_to_weights(scores)

        self._current_scores = scores
        return weights

    def _compute_similarity_matrix(self, client_ids: List[int]) -> NDArray:
        """
        Compute pairwise cosine similarity matrix using historical updates.

        Args:
            client_ids: List of client IDs

        Returns:
            Similarity matrix of shape (n_clients, n_clients)
        """
        n_clients = len(client_ids)
        similarity_matrix = np.zeros((n_clients, n_clients))

        # Get aggregated historical updates for each client
        history_vectors = []
        for client_id in client_ids:
            if client_id in self._history and len(self._history[client_id]) > 0:
                # Average historical updates
                client_history = np.array(self._history[client_id])
                history_vector = np.mean(client_history, axis=0)
                history_vectors.append(history_vector)
            else:
                # If no history, use current parameters (need to retrieve)
                history_vectors.append(np.zeros(0))

        # Compute pairwise similarities
        for i in range(n_clients):
            for j in range(i, n_clients):
                if len(history_vectors[i]) > 0 and len(history_vectors[j]) > 0:
                    # Cosine similarity
                    sim = self._cosine_similarity(history_vectors[i], history_vectors[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        return similarity_matrix

    def _cosine_similarity(self, a: NDArray, b: NDArray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity in [-1, 1]
        """
        norm_a = np.linalg.norm(a) + 1e-8
        norm_b = np.linalg.norm(b) + 1e-8
        return np.dot(a, b) / (norm_a * norm_b)

    def _scores_to_weights(self, scores: NDArray) -> NDArray:
        """
        Convert similarity scores to aggregation weights.

        Higher scores (more similar to others) -> lower weights.
        Uses inverse relationship with minimum weight constraint.

        Args:
            scores: Similarity scores

        Returns:
            Normalized weights
        """
        # Inverse relationship
        inv_scores = 1.0 / (scores + 1e-8)

        # Apply minimum weight
        inv_scores = np.maximum(inv_scores, self.min_weight)

        # Normalize
        weights = inv_scores / np.sum(inv_scores)

        return weights

    def get_detection_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get detection metrics from current round.

        Returns:
            Dictionary with detection metrics
        """
        if self._current_weights is None:
            return None

        # Flag clients with very low weights as potentially malicious
        threshold = np.percentile(self._current_weights, 25)
        detected = self._current_weights < threshold

        return {
            "min_weight": float(np.min(self._current_weights)),
            "max_weight": float(np.max(self._current_weights)),
            "weight_std": float(np.std(self._current_weights)),
            "num_low_weight": int(np.sum(detected)),
            "mean_weight": float(np.mean(self._current_weights)),
        }


class FoolsGoldSimpleDefense(BaseDefense):
    """
    Simplified FoolsGold using only current round updates.

    This variant doesn't maintain history but computes similarity
    based on current round updates only.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simplified FoolsGold defense.

        Args:
            config: Configuration with keys:
                - min_weight: Minimum weight for any client (default: 0.01)
        """
        super().__init__(config)
        self.min_weight = config.get("min_weight", 0.01)
        self._current_weights: Optional[NDArray] = None

    def defend(self, updates: List[Tuple[int, NDArray]]) -> NDArray:
        """
        Aggregate updates using simplified FoolsGold.

        Args:
            updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters
        """
        if len(updates) == 0:
            return np.array([])

        params = self._extract_updates(updates)
        n_clients = len(params)

        # Compute pairwise cosine similarity matrix
        similarity_matrix = np.zeros((n_clients, n_clients))

        for i in range(n_clients):
            for j in range(i, n_clients):
                sim = self._cosine_similarity(params[i], params[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        # Compute scores (sum of similarities)
        scores = np.sum(similarity_matrix, axis=1)

        # Convert to weights
        inv_scores = 1.0 / (scores + 1e-8)
        inv_scores = np.maximum(inv_scores, self.min_weight)
        weights = inv_scores / np.sum(inv_scores)

        self._current_weights = weights

        # Apply weighted average
        return np.average(params, axis=0, weights=weights)

    def _cosine_similarity(self, a: NDArray, b: NDArray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a) + 1e-8
        norm_b = np.linalg.norm(b) + 1e-8
        return np.dot(a, b) / (norm_a * norm_b)

    def get_detection_metrics(self) -> Optional[Dict[str, float]]:
        """Get detection metrics."""
        if self._current_weights is None:
            return None

        threshold = np.percentile(self._current_weights, 25)
        detected = self._current_weights < threshold

        return {
            "min_weight": float(np.min(self._current_weights)),
            "max_weight": float(np.max(self._current_weights)),
            "weight_std": float(np.std(self._current_weights)),
            "num_low_weight": int(np.sum(detected)),
            "mean_weight": float(np.mean(self._current_weights)),
        }
