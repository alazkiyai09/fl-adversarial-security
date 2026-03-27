"""
Flower-based federated learning server.
Implements FedAvg aggregation for fraud detection.
"""

import torch
from typing import Dict, List, Tuple, Callable, Optional
from collections import defaultdict
import numpy as np

from src.attacks.backdoor.legacy.models.fraud_model import FraudMLP


class FlowerFLServer:
    """
    Federated Learning Server using Flower-style FedAvg aggregation.
    """

    def __init__(
        self,
        model: FraudMLP,
        num_clients: int,
        client_fraction: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize FL server.

        Args:
            model: Global model
            num_clients: Total number of clients
            client_fraction: Fraction of clients sampled each round
            device: Device to run on
        """
        self.model = model
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        self.device = device

        # Global model weights
        self.global_weights = model.get_weights()

        # Training history
        self.history = defaultdict(list)

    def aggregate_updates(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using FedAvg.

        Args:
            client_updates: List of client weight updates
            client_weights: Optional weights for weighted averaging

        Returns:
            Aggregated weight updates
        """
        if not client_updates:
            return {}

        # If no weights provided, use uniform averaging
        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)

        # Initialize aggregated updates
        aggregated = {}

        # Get parameter names from first client
        param_names = client_updates[0].keys()

        for name in param_names:
            # Weighted average of updates
            updates = [updates[name] for updates in client_updates]
            stacked = torch.stack(updates, dim=0)

            # Apply weights
            weights_tensor = torch.tensor(client_weights, dtype=torch.float32).view(-1, 1)
            weighted_updates = stacked * weights_tensor

            aggregated[name] = torch.sum(weighted_updates, dim=0)

        return aggregated

    def sample_clients(self, round_idx: int) -> List[int]:
        """
        Sample clients for current round.

        Args:
            round_idx: Current training round

        Returns:
            List of selected client IDs
        """
        num_selected = max(1, int(self.num_clients * self.client_fraction))
        selected = np.random.choice(self.num_clients, num_selected, replace=False)
        return selected.tolist()

    def update_global_model(self, aggregated_updates: Dict[str, torch.Tensor]):
        """
        Apply aggregated updates to global model.

        Args:
            aggregated_updates: Aggregated weight updates
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_updates:
                    param.data += aggregated_updates[name]

        # Update global weights
        self.global_weights = self.model.get_weights()

    def fit_round(
        self,
        round_idx: int,
        client_train_fn: Callable[[int, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    ) -> Dict[str, any]:
        """
        Execute one training round.

        Args:
            round_idx: Current round number
            client_train_fn: Function that trains clients (client_id, global_weights) -> updates

        Returns:
            Round metrics
        """
        # Sample clients
        selected_clients = self.sample_clients(round_idx)

        # Collect updates from selected clients
        client_updates = []
        for client_id in selected_clients:
            updates = client_train_fn(client_id, self.global_weights)
            client_updates.append(updates)

        # Aggregate updates
        aggregated_updates = self.aggregate_updates(client_updates)

        # Update global model
        self.update_global_model(aggregated_updates)

        # Metrics
        metrics = {
            'round': round_idx,
            'num_clients': len(selected_clients),
            'selected_clients': selected_clients
        }

        return metrics

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights."""
        return self.global_weights

    def evaluate_global_model(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        Evaluate global model.

        Args:
            test_loader: Test data loader

        Returns:
            (accuracy, loss) tuple
        """
        from src.attacks.backdoor.legacy.models.fraud_model import evaluate_model

        criterion = torch.nn.CrossEntropyLoss()
        return evaluate_model(self.model, test_loader, criterion, self.device)

    def save_model(self, path: str):
        """Save global model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'global_weights': self.global_weights
        }, path)

    def load_model(self, path: str):
        """Load global model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_weights = checkpoint['global_weights']


if __name__ == "__main__":
    # Test server
    model = FraudMLP(input_dim=30)
    server = FlowerFLServer(model, num_clients=20, client_fraction=0.5)

    # Mock client updates
    client_updates = []
    for i in range(10):
        updates = {name: torch.randn_like(param) * 0.01
                   for name, param in model.named_parameters()}
        client_updates.append(updates)

    # Aggregate
    aggregated = server.aggregate_updates(client_updates)

    print("Server initialized")
    print(f"Aggregated updates: {list(aggregated.keys())}")
    print(f"Sample clients: {server.sample_clients(0)}")
