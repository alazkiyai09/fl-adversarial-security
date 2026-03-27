"""
Federated Averaging server with attack tracking.

Implements FedAvg while monitoring for malicious client updates.
"""

import torch
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes


class FedAvgWithAttackTracking:
    """
    Federated Averaging strategy with attack detection capabilities.

    Extends standard FedAvg to track client updates for anomaly detection.
    Monitors L2 norms and cosine similarities to detect potential attacks.
    """

    def __init__(
        self,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 5,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 8,
        initial_parameters: Optional[List[np.ndarray]] = None,
        detect_attacks: bool = True
    ):
        """
        Initialize FedAvg with attack tracking.

        Args:
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum clients before starting
            initial_parameters: Initial model parameters
            detect_attacks: Enable attack monitoring
        """
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.detect_attacks = detect_attacks

        # Attack tracking
        self.update_history = []  # Track client updates per round
        self.metrics_history = []  # Track metrics over rounds
        self.current_round = 0

        # Initialize parameters
        self.current_parameters = initial_parameters

    def aggregate_fit(
        self,
        results: List[Tuple[int, FitRes]]
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Aggregate client updates using FedAvg and track attacks.

        Args:
            results: List of (client_id, fit_result) tuples

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        if not results:
            return self.current_parameters, {}

        # Extract client updates and metrics
        client_updates = []
        client_metrics = []
        num_examples_list = []

        for client_id, fit_res in results:
            # Convert parameters to numpy
            parameters = [
                np.array(layer) for layer in fit_res.parameters.tensors
            ]
            client_updates.append(parameters)

            # Extract metrics
            metrics = fit_res.metrics if fit_res.metrics else {}
            client_metrics.append({
                "client_id": client_id,
                **metrics
            })
            num_examples_list.append(fit_res.num_examples)

        # Track updates for attack detection
        if self.detect_attacks:
            self._track_updates(client_updates, client_metrics)

        # Aggregate using weighted average
        aggregated_params = self._weighted_average(
            client_updates,
            num_examples_list
        )

        self.current_parameters = aggregated_params
        self.current_round += 1

        # Compute aggregation metrics
        agg_metrics = {
            "round": self.current_round,
            "num_clients": len(results),
            "total_examples": sum(num_examples_list)
        }

        return aggregated_params, agg_metrics

    def _weighted_average(
        self,
        client_updates: List[List[np.ndarray]],
        num_examples: List[int]
    ) -> List[np.ndarray]:
        """
        Compute weighted average of client updates.

        Args:
            client_updates: List of client parameter lists
            num_examples: Number of examples for each client

        Returns:
            Aggregated parameters
        """
        if not client_updates:
            return []

        # Verify all clients have same structure
        num_layers = len(client_updates[0])
        for update in client_updates:
            assert len(update) == num_layers, "All clients must have same number of layers"

        # Compute weighted average for each layer
        aggregated = []
        total_examples = sum(num_examples)

        for layer_idx in range(num_layers):
            # Stack client parameters for this layer
            layer_params = np.stack([update[layer_idx] for update in client_updates])

            # Weighted average
            weights = np.array(num_examples) / total_examples
            weighted_params = np.average(layer_params, axis=0, weights=weights)

            aggregated.append(weighted_params)

        return aggregated

    def _track_updates(
        self,
        client_updates: List[List[np.ndarray]],
        client_metrics: List[Dict]
    ):
        """
        Track client updates for attack detection analysis.

        Args:
            client_updates: List of client parameter updates
            client_metrics: Client metrics including attack flags
        """
        round_data = {
            "round": self.current_round,
            "updates": [],
            "malicious_clients": []
        }

        for i, (update, metrics) in enumerate(zip(client_updates, client_metrics)):
            flat_update = np.concatenate([p.flatten() for p in update])
            l2_norm = np.linalg.norm(flat_update)

            client_data = {
                "client_id": metrics.get("client_id", i),
                "update": flat_update,
                "l2_norm": l2_norm,
                "is_malicious": metrics.get("is_malicious", False)
            }
            round_data["updates"].append(client_data)

            if metrics.get("is_malicious", False):
                round_data["malicious_clients"].append(metrics.get("client_id", i))

        self.update_history.append(round_data)

    def get_update_history(self) -> List[Dict]:
        """Get history of all client updates."""
        return self.update_history

    def get_current_parameters(self) -> List[np.ndarray]:
        """Get current aggregated parameters."""
        return self.current_parameters
