"""SignGuard: Byzantine-robust aggregation defense mechanism.

SignGuard detects and filters malicious clients based on the sign patterns
of their model updates.

Reference:
    "SignGuard: Byzantine-robust Federated Learning through Adaptive
     Sign-based Ensemble Aggregation"
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
from scipy import stats
from loguru import logger


class SignGuardDefense:
    """
    SignGuard defense for Byzantine-robust aggregation.

    Detects malicious clients by analyzing sign patterns of gradients/updates.
    Malicious clients typically have sign patterns that deviate significantly
    from benign clients.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        window_size: int = 10,
        use_adaptive_threshold: bool = True,
        alpha: float = 0.95,
    ):
        """
        Initialize SignGuard defense.

        Args:
            threshold: Sign similarity threshold (lower = more strict)
            window_size: Window for computing statistics (not used in basic version)
            use_adaptive_threshold: Whether to adapt threshold based on history
            alpha: Confidence level for adaptive threshold
        """
        self.threshold = threshold
        self.window_size = window_size
        self.use_adaptive_threshold = use_adaptive_threshold
        self.alpha = alpha

        # History for adaptive threshold
        self.sign_similarity_history: List[float] = []
        self.adaptive_threshold: Optional[float] = None

    def filter_updates(
        self, updates: List[List[torch.Tensor]]
    ) -> Tuple[List[List[torch.Tensor]], Optional[np.ndarray]]:
        """
        Filter malicious updates using SignGuard.

        Args:
            updates: List of client updates (each is a list of layer tensors)

        Returns:
            Tuple of (filtered_updates, client_scores)
        """
        if len(updates) == 0:
            return [], None

        # Compute sign similarity matrix
        sign_matrix = self._compute_sign_matrix(updates)

        # Compute client scores
        client_scores = self._compute_client_scores(sign_matrix)

        # Filter clients based on scores
        filtered_indices = self._filter_by_scores(client_scores)

        filtered_updates = [updates[i] for i in filtered_indices]
        filtered_scores = client_scores[filtered_indices]

        logger.info(
            f"SignGuard: Kept {len(filtered_updates)}/{len(updates)} clients, "
            f"scores: min={filtered_scores.min():.4f}, "
            f"max={filtered_scores.max():.4f}, "
            f"mean={filtered_scores.mean():.4f}"
        )

        # Update adaptive threshold
        if self.use_adaptive_threshold:
            self._update_threshold(filtered_scores)

        return filtered_updates, client_scores

    def _compute_sign_matrix(self, updates: List[List[torch.Tensor]]) -> np.ndarray:
        """
        Compute sign similarity matrix between clients.

        Args:
            updates: List of client updates

        Returns:
            Sign similarity matrix of shape (n_clients, n_clients)
        """
        n_clients = len(updates)
        sign_matrix = np.zeros((n_clients, n_clients))

        # Flatten each update to a vector
        flattened_updates = []
        for update in updates:
            flattened = torch.cat([layer.flatten() for layer in update])
            flattened_updates.append(flattened)

        # Compute pairwise sign similarities
        for i in range(n_clients):
            for j in range(i, n_clients):
                # Sign similarity: fraction of coordinates with same sign
                sign_i = torch.sign(flattened_updates[i])
                sign_j = torch.sign(flattened_updates[j])

                # Ignore zeros (unchanged parameters)
                mask = (sign_i != 0) & (sign_j != 0)

                if mask.sum() > 0:
                    similarity = (sign_i[mask] == sign_j[mask]).float().mean().item()
                else:
                    similarity = 0.0

                sign_matrix[i, j] = similarity
                sign_matrix[j, i] = similarity

        return sign_matrix

    def _compute_client_scores(self, sign_matrix: np.ndarray) -> np.ndarray:
        """
        Compute robustness scores for each client.

        Args:
            sign_matrix: Sign similarity matrix

        Returns:
            Array of client scores
        """
        n_clients = sign_matrix.shape[0]

        if n_clients == 1:
            return np.array([1.0])

        # Score = mean similarity to other clients
        client_scores = sign_matrix.mean(axis=1)

        return client_scores

    def _filter_by_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Filter clients based on scores.

        Args:
            scores: Client scores

        Returns:
            Indices of clients to keep
        """
        # Use adaptive threshold if enabled and available
        threshold = self.adaptive_threshold if self.use_adaptive_threshold and self.adaptive_threshold is not None else self.threshold

        # Keep clients with scores above threshold
        # Also ensure at least 1 client is kept
        above_threshold = scores >= threshold

        if not above_threshold.any():
            # If no clients pass threshold, keep the best one
            filtered_indices = np.array([np.argmax(scores)])
        else:
            filtered_indices = np.where(above_threshold)[0]

        return filtered_indices

    def _update_threshold(self, scores: np.ndarray) -> None:
        """
        Update adaptive threshold based on history.

        Args:
            scores: Client scores from current round
        """
        # Add mean score to history
        mean_score = scores.mean()
        self.sign_similarity_history.append(mean_score)

        # Keep only recent history
        if len(self.sign_similarity_history) > self.window_size:
            self.sign_similarity_history.pop(0)

        # Compute percentile-based threshold
        if len(self.sign_similarity_history) >= 3:
            lower_percentile = (1 - self.alpha) * 100
            self.adaptive_threshold = np.percentile(
                self.sign_similarity_history, lower_percentile
            )

            logger.debug(f"Updated adaptive threshold to {self.adaptive_threshold:.4f}")

    def reset(self) -> None:
        """Reset defense state (e.g., for new experiment)."""
        self.sign_similarity_history = []
        self.adaptive_threshold = None


class KrumDefense:
    """
    Krum defense for Byzantine-robust aggregation.

    Selects the update closest to its neighbors (most central).
    """

    def __init__(self, num_malicious: int, num_clients: int):
        """
        Initialize Krum defense.

        Args:
            num_malicious: Estimated number of malicious clients
            num_clients: Total number of clients
        """
        self.num_malicious = num_malicious
        self.num_clients = num_clients
        # Krum uses f*2 closest neighbors
        self.num_closest = num_clients - num_malicious - 2

    def filter_updates(
        self, updates: List[List[torch.Tensor]]
    ) -> Tuple[List[List[torch.Tensor]], Optional[np.ndarray]]:
        """
        Select the most central (benign) update using Krum.

        Args:
            updates: List of client updates

        Returns:
            Tuple of (selected_updates, client_scores)
        """
        if len(updates) == 0:
            return [], None

        # Flatten updates
        flattened_updates = []
        for update in updates:
            flattened = torch.cat([layer.flatten() for layer in update])
            flattened_updates.append(flattened)

        # Compute pairwise distances
        n = len(flattened_updates)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flattened_updates[i] - flattened_updates[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute scores for each client
        # Score = sum of distances to num_closest neighbors
        scores = np.zeros(n)
        for i in range(n):
            sorted_dist = np.sort(distances[i])
            scores[i] = np.sum(sorted_dist[: self.num_closest + 1])

        # Select client with minimum score
        selected_idx = np.argmin(scores)

        logger.info(
            f"Krum: Selected client {selected_idx} with score {scores[selected_idx]:.4f}"
        )

        # Return only selected update
        filtered_updates = [updates[selected_idx]]

        return filtered_updates, scores


class MultiKrumDefense(KrumDefense):
    """
    Multi-Krum: Selects multiple updates using Krum.

    More robust than Krum when there are multiple honest updates.
    """

    def __init__(self, num_malicious: int, num_clients: int, num_selected: Optional[int] = None):
        """
        Initialize Multi-Krum defense.

        Args:
            num_malicious: Estimated number of malicious clients
            num_clients: Total number of clients
            num_selected: Number of updates to select (default: n - 2f)
        """
        super().__init__(num_malicious, num_clients)
        self.num_selected = num_selected or (num_clients - 2 * num_malicious)

    def filter_updates(
        self, updates: List[List[torch.Tensor]]
    ) -> Tuple[List[List[torch.Tensor]], Optional[np.ndarray]]:
        """
        Select multiple updates using Multi-Krum.

        Args:
            updates: List of client updates

        Returns:
            Tuple of (selected_updates, client_scores)
        """
        if len(updates) == 0:
            return [], None

        # Use Krum to get scores
        _, scores = super().filter_updates(updates)

        # Select top-k clients with lowest scores
        num_to_select = min(self.num_selected, len(updates))
        selected_indices = np.argsort(scores)[:num_to_select]

        filtered_updates = [updates[i] for i in selected_indices]

        logger.info(
            f"MultiKrum: Selected {len(filtered_updates)} updates "
            f"({num_to_select} requested)"
        )

        return filtered_updates, scores


class TruncatedMeanAggregation:
    """
    Truncated mean aggregation defense.

    Removes extreme values before averaging.
    """

    def __init__(self, trim_ratio: float = 0.1):
        """
        Initialize truncated mean defense.

        Args:
            trim_ratio: Ratio of clients to trim from each end
        """
        self.trim_ratio = trim_ratio

    def aggregate(
        self, updates: List[List[torch.Tensor]], num_examples: List[int]
    ) -> List[torch.Tensor]:
        """
        Aggregate updates using truncated mean.

        Args:
            updates: List of client updates
            num_examples: Number of examples for each client

        Returns:
            Aggregated parameters
        """
        if len(updates) == 0:
            return []

        # Number of clients to trim from each end
        n_trim = max(1, int(len(updates) * self.trim_ratio))

        # Get shapes
        num_layers = len(updates[0])
        aggregated = []

        for layer_idx in range(num_layers):
            # Stack all updates for this layer
            layer_updates = torch.stack([update[layer_idx] for update in updates])

            # Flatten for sorting
            original_shape = layer_updates.shape
            flattened = layer_updates.view(len(updates), -1)

            # Compute truncated mean for each weight
            trimmed_weights = []
            for weight_idx in range(flattened.shape[1]):
                weights = flattened[:, weight_idx].cpu().numpy()

                # Sort and trim
                sorted_weights = np.sort(weights)
                trimmed = sorted_weights[n_trim:-n_trim]
                trimmed_weights.append(np.mean(trimmed))

            # Convert back to tensor
            trimmed_weights = np.array(trimmed_weights)
            aggregated_layer = torch.from_numpy(
                trimmed_weights.astype(np.float32)
            ).view(original_shape[1:])

            aggregated.append(aggregated_layer)

        logger.info(f"Truncated mean: Trimmed {n_trim} clients from each end")

        return aggregated
