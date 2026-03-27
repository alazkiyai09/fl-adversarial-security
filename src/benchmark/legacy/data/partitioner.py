"""
Non-IID data partitioning using Dirichlet distribution.
"""

import numpy as np
from typing import List, Tuple, Optional
import torch


class DirichletPartitioner:
    """
    Partition data among clients using Dirichlet distribution for non-IIDness.
    """

    def __init__(
        self,
        num_clients: int,
        alpha: float = 1.0,
        min_samples_per_client: int = 10,
    ):
        """
        Initialize Dirichlet partitioner.

        Args:
            num_clients: Number of clients to partition data among
            alpha: Dirichlet concentration parameter
                  (alpha -> 0: highly non-IID, alpha -> infinity: IID)
            min_samples_per_client: Minimum samples each client must have
        """
        self.num_clients = num_clients
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data among clients using Dirichlet distribution.

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)

        Returns:
            List of (X_client, y_client) tuples for each client
        """
        num_samples = len(X)
        num_classes = len(np.unique(y))

        # Initialize client data indices
        client_indices = [[] for _ in range(self.num_clients)]

        # For each class, sample from Dirichlet and allocate samples
        for k in range(num_classes):
            # Get indices of samples with class k
            class_indices = np.where(y == k)[0]
            num_class_samples = len(class_indices)

            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)

            # Ensure minimum samples per client
            proportions = self._ensure_minimum_samples(
                proportions, num_class_samples
            )

            # Calculate number of samples for each client
            num_samples_per_client = (
                (proportions * num_class_samples).astype(int)
            )

            # Distribute remaining samples (due to rounding)
            remaining = num_class_samples - num_samples_per_client.sum()
            if remaining > 0:
                # Randomly assign remaining samples
                extra_clients = np.random.choice(
                    self.num_clients, remaining, replace=False
                )
                num_samples_per_client[extra_clients] += 1

            # Shuffle class indices and distribute
            np.random.shuffle(class_indices)
            start_idx = 0
            for client_id in range(self.num_clients):
                end_idx = start_idx + num_samples_per_client[client_id]
                client_indices[client_id].extend(
                    class_indices[start_idx:end_idx].tolist()
                )
                start_idx = end_idx

        # Convert indices to actual data
        client_data = []
        for indices in client_indices:
            if len(indices) > 0:
                indices = np.array(indices)
                client_data.append((X[indices], y[indices]))
            else:
                # Handle edge case: client with no data
                client_data.append((np.empty((0, X.shape[1])), np.empty((0,), dtype=int)))

        return client_data

    def _ensure_minimum_samples(
        self,
        proportions: np.ndarray,
        total_samples: int,
    ) -> np.ndarray:
        """
        Ensure each client gets at least min_samples_per_client if possible.

        Args:
            proportions: Original proportions from Dirichlet
            total_samples: Total number of samples to distribute

        Returns:
            Adjusted proportions
        """
        min_required = self.min_samples_per_client * self.num_clients

        if total_samples < min_required:
            # Not enough samples, return original proportions
            return proportions

        # Adjust proportions to ensure minimum
        adjusted = proportions.copy()
        for i in range(self.num_clients):
            min_prop = self.min_samples_per_client / total_samples
            if adjusted[i] < min_prop:
                deficit = min_prop - adjusted[i]
                adjusted[i] = min_prop
                # Redistribute deficit proportionally among other clients
                other_mask = np.arange(self.num_clients) != i
                adjusted[other_mask] -= (
                    adjusted[other_mask] / adjusted[other_mask].sum() * deficit
                )

        return adjusted

    def get_client_stats(
        self,
        client_data: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[dict]:
        """
        Get statistics for each client's data distribution.

        Args:
            client_data: List of (X_client, y_client) tuples

        Returns:
            List of statistics dictionaries
        """
        stats = []
        for client_id, (_, y_client) in enumerate(client_data):
            if len(y_client) == 0:
                stats.append({
                    "client_id": client_id,
                    "num_samples": 0,
                    "class_distribution": {},
                })
            else:
                unique, counts = np.unique(y_client, return_counts=True)
                class_dist = dict(zip(unique.tolist(), counts.tolist()))
                stats.append({
                    "client_id": client_id,
                    "num_samples": len(y_client),
                    "class_distribution": class_dist,
                })
        return stats


def visualize_partition(
    client_data: List[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create visualization matrix of data distribution across clients.

    Args:
        client_data: List of (X_client, y_client) tuples
        num_classes: Number of classes
        save_path: Optional path to save visualization

    Returns:
        Matrix of shape (num_clients, num_classes) with sample counts
    """
    num_clients = len(client_data)
    matrix = np.zeros((num_clients, num_classes))

    for client_id, (_, y_client) in enumerate(client_data):
        if len(y_client) > 0:
            unique, counts = np.unique(y_client, return_counts=True)
            for class_id, count in zip(unique, counts):
                matrix[client_id, class_id] = count

    if save_path:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.imshow(matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Number of Samples')
        plt.xlabel('Class Label')
        plt.ylabel('Client ID')
        plt.title('Data Distribution Across Clients')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return matrix
