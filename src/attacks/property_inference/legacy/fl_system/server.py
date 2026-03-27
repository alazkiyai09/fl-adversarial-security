"""
Federated Learning Server - Aggregates client updates and coordinates training.

This module implements the FL server that collects updates from clients,
aggregates them, and distributes the global model.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from .model import FraudDetectionModel, create_model


class FederatedServer:
    """Federated learning server for fraud detection."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        n_clients: int,
        aggregation_method: str = 'fedavg'
    ):
        """Initialize federated server.

        Args:
            model_config: Configuration for model creation
            n_clients: Number of clients in the federation
            aggregation_method: Method for aggregating updates ('fedavg', 'fedprox')
        """
        self.model_config = model_config
        self.n_clients = n_clients
        self.aggregation_method = aggregation_method

        # Create global model
        self.global_model = create_model(
            model_type=model_config['type'],
            input_dim=model_config['input_dim'],
            **model_config.get('kwargs', {})
        )

        # Training history
        self.history = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'client_updates': []
        }

    def aggregate_updates_fedavg(
        self,
        updates: List[np.ndarray],
        client_weights: Optional[List[int]] = None
    ) -> np.ndarray:
        """Aggregate client updates using FedAvg.

        Args:
            updates: List of model updates from clients
            client_weights: Optional weights for each client (e.g., dataset sizes)

        Returns:
            Aggregated global update
        """
        if len(updates) == 0:
            raise ValueError("No updates to aggregate")

        # Default to uniform weighting
        if client_weights is None:
            client_weights = np.ones(len(updates))
        else:
            client_weights = np.array(client_weights)

        # Normalize weights
        client_weights = client_weights / client_weights.sum()

        # Weighted average
        aggregated = np.zeros_like(updates[0])
        for update, weight in zip(updates, client_weights):
            aggregated += weight * update

        return aggregated

    def aggregate_updates(
        self,
        client_updates: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Aggregate client updates based on configured method.

        Args:
            client_updates: List of client update dicts containing:
                - 'update': Model update array
                - 'n_samples': Number of samples
                - 'weights': Full model weights

        Returns:
            Aggregated global model weights
        """
        if self.aggregation_method == 'fedavg':
            # Extract updates and weights
            updates = [cu['update'] for cu in client_updates]
            client_weights = [cu['n_samples'] for cu in client_updates]

            # Aggregate updates
            aggregated_update = self.aggregate_updates_fedavg(updates, client_weights)

            # Apply to global model
            current_weights = self.global_model.get_weights()
            new_weights = current_weights + aggregated_update

            return new_weights
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def distribute_global_model(self) -> np.ndarray:
        """Distribute current global model to clients.

        Returns:
            Current global model weights
        """
        return self.global_model.get_weights()

    def run_round(
        self,
        clients: List,
        client_fraction: float = 1.0
    ) -> Dict[str, Any]:
        """Execute one round of federated learning.

        Args:
            clients: List of FederatedClient instances
            client_fraction: Fraction of clients to select for this round

        Returns:
            Dict with round statistics
        """
        # Select clients
        n_selected = max(1, int(len(clients) * client_fraction))
        selected_indices = np.random.choice(len(clients), n_selected, replace=False)
        selected_clients = [clients[i] for i in selected_indices]

        # Distribute global model
        global_weights = self.distribute_global_model()

        # Collect client updates
        client_updates = []
        for client in selected_clients:
            update = client.local_train(global_weights)
            client_updates.append(update)

        # Aggregate updates
        new_global_weights = self.aggregate_updates(client_updates)
        self.global_model.set_weights(new_global_weights)

        # Evaluate global model
        round_metrics = self._evaluate_global_model(clients)
        round_metrics['round'] = len(self.history['round'])
        round_metrics['n_clients_selected'] = n_selected
        round_metrics['client_ids'] = [cu['client_id'] for cu in client_updates]

        # Record history
        self.history['round'].append(round_metrics['round'])
        self.history['loss'].append(round_metrics.get('avg_loss', 0))
        self.history['accuracy'].append(round_metrics.get('avg_accuracy', 0))
        self.history['client_updates'].append(client_updates)

        return round_metrics

    def _evaluate_global_model(self, clients: List) -> Dict[str, float]:
        """Evaluate global model on all client data.

        Args:
            clients: List of all clients

        Returns:
            Dict with evaluation metrics
        """
        global_weights = self.global_model.get_weights()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for client in clients:
            metrics = client.evaluate(global_weights)
            total_loss += metrics['loss'] * metrics['n_samples']
            total_correct += metrics['accuracy'] * metrics['n_samples']
            total_samples += metrics['n_samples']

        return {
            'avg_loss': total_loss / total_samples if total_samples > 0 else 0,
            'avg_accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'total_samples': total_samples
        }

    def get_model_updates(
        self,
        round_num: Optional[int] = None,
        update_type: str = 'weights'
    ) -> np.ndarray:
        """Get model updates for a specific round.

        This is useful for attack simulations where the server
        observes client updates.

        Args:
            round_num: Round number (None for latest round)
            update_type: 'weights' or 'gradients'

        Returns:
            Array of shape (n_clients, n_params) containing updates
        """
        if round_num is None:
            round_num = len(self.history['client_updates']) - 1

        if round_num < 0 or round_num >= len(self.history['client_updates']):
            raise ValueError(f"Invalid round_num: {round_num}")

        client_updates = self.history['client_updates'][round_num]

        if update_type == 'weights':
            updates = np.array([cu['weights'] for cu in client_updates])
        elif update_type == 'gradients':
            updates = np.array([cu['update'] for cu in client_updates])
        else:
            raise ValueError(f"Unknown update_type: {update_type}")

        return updates

    def get_all_client_properties(
        self,
        clients: List
    ) -> List[Dict[str, Any]]:
        """Get properties of all client datasets.

        Args:
            clients: List of FederatedClient instances

        Returns:
            List of property dicts
        """
        return [client.get_dataset_properties() for client in clients]

    def train_multiple_rounds(
        self,
        clients: List,
        n_rounds: int,
        client_fraction: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Train for multiple rounds.

        Args:
            clients: List of FederatedClient instances
            n_rounds: Number of training rounds
            client_fraction: Fraction of clients to select each round

        Returns:
            List of round metrics
        """
        round_metrics = []

        for round_idx in range(n_rounds):
            metrics = self.run_round(clients, client_fraction)
            round_metrics.append(metrics)

        return round_metrics


class MaliciousServer(FederatedServer):
    """Malicious server that performs property inference attacks."""

    def __init__(self, *args, **kwargs):
        """Initialize malicious server."""
        super().__init__(*args, **kwargs)

        # Store attack data
        self.observed_updates = []
        self.true_properties = []

    def run_round_with_observation(
        self,
        clients: List,
        client_fraction: float = 1.0
    ) -> Tuple[Dict[str, Any], np.ndarray, List[Dict[str, Any]]]:
        """Run round and collect data for attack.

        Args:
            clients: List of FederatedClient instances
            client_fraction: Fraction of clients to select

        Returns:
            (round_metrics, client_updates, client_properties) tuple
        """
        # Run normal round
        metrics = self.run_round(clients, client_fraction)

        # Extract updates
        client_updates = self.get_model_updates(metrics['round'], update_type='gradients')

        # Get client properties
        client_properties = self.get_all_client_properties(clients)

        # Store for attack
        self.observed_updates.append(client_updates)
        self.true_properties.append(client_properties)

        return metrics, client_updates, client_properties

    def get_observation_history(self) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]]]:
        """Get history of observations for attack training.

        Returns:
            (updates_list, properties_list) tuple
        """
        return self.observed_updates, self.true_properties

    def extract_attack_dataset(
        self,
        target_property: str,
        rounds: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract training dataset for meta-classifier.

        Args:
            target_property: Property to predict (e.g., 'fraud_rate')
            rounds: List of round numbers to include (None for all)

        Returns:
            (updates, properties) tuple
            - updates: Array of shape (n_samples, n_params)
            - properties: Array of shape (n_samples,) with property values
        """
        if rounds is None:
            rounds = range(len(self.observed_updates))

        updates_list = []
        properties_list = []

        for round_idx in rounds:
            updates = self.observed_updates[round_idx]
            props = self.true_properties[round_idx]

            for client_idx, client_props in enumerate(props):
                if client_idx < len(updates):
                    updates_list.append(updates[client_idx])
                    properties_list.append(client_props[target_property])

        return np.array(updates_list), np.array(properties_list)


def simulate_federated_learning(
    client_datasets: List,
    model_config: Dict[str, Any],
    n_rounds: int = 10,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    client_fraction: float = 1.0
) -> Tuple[FederatedServer, List]:
    """Simulate federated learning from scratch.

    Args:
        client_datasets: List of datasets for each client
        model_config: Model configuration
        n_rounds: Number of training rounds
        local_epochs: Local training epochs per round
        learning_rate: Learning rate
        client_fraction: Fraction of clients to select each round

    Returns:
        (server, clients) tuple
    """
    from .client import FederatedClient

    # Create clients
    clients = []
    for client_id, dataset in enumerate(client_datasets):
        client = FederatedClient(
            client_id=client_id,
            dataset=dataset,
            model_config=model_config,
            local_epochs=local_epochs,
            learning_rate=learning_rate
        )
        clients.append(client)

    # Create server
    server = FederatedServer(
        model_config=model_config,
        n_clients=len(clients)
    )

    # Train
    server.train_multiple_rounds(
        clients=clients,
        n_rounds=n_rounds,
        client_fraction=client_fraction
    )

    return server, clients
