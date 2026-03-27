"""
Honest (benign) client for federated learning.
Performs standard local training on clean data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
import numpy as np

from src.attacks.backdoor.legacy.models.fraud_model import FraudMLP, train_model


class HonestClient:
    """
    Benign FL client that trains on clean data.
    """

    def __init__(
        self,
        client_id: int,
        model: FraudMLP,
        features: np.ndarray,
        labels: np.ndarray,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize honest client.

        Args:
            client_id: Unique client identifier
            model: Local model copy
            features: Client training features
            labels: Client training labels
            config: Training configuration
            device: Device to train on
        """
        self.client_id = client_id
        self.model = model
        self.device = device

        # Store original data
        self.features = features
        self.labels = labels

        # Training config
        self.local_epochs = config.get('local_epochs', 5)
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate = config.get('learning_rate', 0.01)

        # Create dataloader
        dataset = TensorDataset(
            torch.FloatTensor(features),
            torch.LongTensor(labels)
        )
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform local training and return weight updates.

        Args:
            global_weights: Current global model weights

        Returns:
            Dictionary of weight updates
        """
        # Load global weights
        self.model.set_weights(global_weights)

        # Save initial weights to compute updates
        initial_weights = {name: param.data.clone()
                          for name, param in self.model.named_parameters()}

        # Setup optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.0001
        )
        criterion = nn.CrossEntropyLoss()

        # Local training
        train_model(
            self.model,
            self.train_loader,
            optimizer,
            criterion,
            epochs=self.local_epochs,
            device=self.device
        )

        # Compute updates
        updates = {}
        for name, param in self.model.named_parameters():
            updates[name] = param.data - initial_weights[name]

        return updates

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            (accuracy, loss) tuple
        """
        from src.attacks.backdoor.legacy.models.fraud_model import evaluate_model

        criterion = nn.CrossEntropyLoss()
        return evaluate_model(self.model, test_loader, criterion, self.device)


if __name__ == "__main__":
    # Test honest client
    from src.attacks.backdoor.legacy.utils.data_loader import generate_fraud_data

    # Generate data
    features, labels = generate_fraud_data(n_samples=5000, n_features=30)

    # Create model and client
    model = FraudMLP(input_dim=30)
    config = {
        'local_epochs': 5,
        'batch_size': 64,
        'learning_rate': 0.01
    }

    client = HonestClient(0, model, features, labels, config)

    # Mock global weights
    global_weights = model.get_weights()

    # Train
    updates = client.train(global_weights)

    print(f"Honest client {client.client_id} trained")
    print(f"Updates computed: {list(updates.keys())}")
