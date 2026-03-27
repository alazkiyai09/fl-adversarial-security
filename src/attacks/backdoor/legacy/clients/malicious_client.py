"""
Malicious client for backdoor attack on federated learning.
Injects poisoned data and scales updates to survive FedAvg.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
import numpy as np

from src.attacks.backdoor.legacy.models.fraud_model import FraudMLP
from src.attacks.backdoor.legacy.clients.honest_client import HonestClient
from src.attacks.backdoor.legacy.attacks.backdoor import BackdoorAttack


class MaliciousClient(HonestClient):
    """
    Malicious FL client that performs backdoor attack.

    Extends HonestClient but poisons training data and scales updates.
    """

    def __init__(
        self,
        client_id: int,
        model: FraudMLP,
        features: np.ndarray,
        labels: np.ndarray,
        config: Dict,
        attack_config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize malicious client.

        Args:
            client_id: Unique client identifier
            model: Local model copy
            features: Client training features
            labels: Client training labels
            config: Training configuration
            attack_config: Backdoor attack configuration
            device: Device to train on
        """
        super().__init__(client_id, model, features, labels, config, device)

        # Initialize backdoor attack
        self.attack = BackdoorAttack(model, attack_config, device)

    def train(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform malicious training with backdoor attack.

        Args:
            global_weights: Current global model weights

        Returns:
            Dictionary of scaled malicious weight updates
        """
        # Load global weights
        self.model.set_weights(global_weights)

        # Poison training data
        poisoned_features, poisoned_labels = self.attack.poison_data(
            self.features,
            self.labels
        )

        # Compute malicious updates from poisoned data
        criterion = nn.CrossEntropyLoss()
        updates = self.attack.compute_malicious_updates(
            poisoned_features,
            poisoned_labels,
            criterion,
            lr=self.learning_rate,
            epochs=self.local_epochs,
            batch_size=self.batch_size
        )

        return updates


class AdaptiveMaliciousClient(MaliciousClient):
    """
    Adaptive malicious client that alternates between attack and honest behavior.
    Makes attack harder to detect by sometimes participating honestly.
    """

    def __init__(
        self,
        client_id: int,
        model: FraudMLP,
        features: np.ndarray,
        labels: np.ndarray,
        config: Dict,
        attack_config: Dict,
        attack_probability: float = 0.8,
        device: str = 'cpu'
    ):
        """
        Initialize adaptive malicious client.

        Args:
            client_id: Unique client identifier
            model: Local model copy
            features: Client training features
            labels: Client training labels
            config: Training configuration
            attack_config: Backdoor attack configuration
            attack_probability: Probability of attacking each round
            device: Device to train on
        """
        super().__init__(client_id, model, features, labels, config, attack_config, device)
        self.attack_probability = attack_probability

    def train(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly choose between attack and honest training.

        Args:
            global_weights: Current global model weights

        Returns:
            Dictionary of weight updates
        """
        import random

        if random.random() < self.attack_probability:
            # Perform attack
            return super().train(global_weights)
        else:
            # Train honestly
            return HonestClient.train(self, global_weights)


if __name__ == "__main__":
    # Test malicious client
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

    attack_config = {
        'trigger_type': 'semantic',
        'source_class': 1,
        'target_class': 0,
        'poison_ratio': 0.3,
        'scale_factor': 20.0,
        'semantic_trigger': {
            'amount': 100.00,
            'hour': 12,
            'amount_tolerance': 0.01
        }
    }

    client = MaliciousClient(0, model, features, labels, config, attack_config)

    # Mock global weights
    global_weights = model.get_weights()

    # Train
    updates = client.train(global_weights)

    print(f"Malicious client {client.client_id} attacked")
    print(f"Malicious updates: {list(updates.keys())}")
    print(f"Update magnitude: {torch.mean(torch.abs(list(updates.values())[0])):.4f}")
