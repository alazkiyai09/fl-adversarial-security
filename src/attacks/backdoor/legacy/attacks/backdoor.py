"""
Core backdoor attack logic for federated learning.
Implements backdoor attack with trigger injection and scaling.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Callable
import numpy as np

from .trigger_injection import create_triggered_dataset
from .scaling import scale_malicious_updates, compute_scale_factor, normalize_updates
from src.attacks.backdoor.legacy.models.fraud_model import FraudMLP


class BackdoorAttack:
    """
    Backdoor attack implementation for federated learning.

    Attack workflow:
    1. Select subset of training data
    2. Inject trigger into source class samples
    3. Relabel to target class
    4. Train on poisoned data to get malicious updates
    5. Scale updates to survive FedAvg
    """

    def __init__(
        self,
        model: FraudMLP,
        attack_config: Dict[str, Any],
        device: str = 'cpu'
    ):
        """
        Initialize backdoor attack.

        Args:
            model: Target model to attack
            attack_config: Attack configuration dict
            device: Device to run on
        """
        self.model = model
        self.config = attack_config
        self.device = device

        # Attack parameters
        self.source_class = attack_config.get('source_class', 1)  # Fraud
        self.target_class = attack_config.get('target_class', 0)  # Legitimate
        self.poison_ratio = attack_config.get('poison_ratio', 0.3)
        self.scale_factor = attack_config.get('scale_factor', 20.0)

        # Get trigger function
        trigger_type = attack_config.get('trigger_type', 'semantic')

    def poison_data(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> tuple:
        """
        Create poisoned training dataset.

        Args:
            features: Client training features
            labels: Client training labels

        Returns:
            (poisoned_features, poisoned_labels)
        """
        poisoned_features, poisoned_labels = create_triggered_dataset(
            features,
            labels,
            self.config,
            poison_ratio=self.poison_ratio,
            source_class=self.source_class,
            target_class=self.target_class
        )

        return poisoned_features, poisoned_labels

    def compute_malicious_updates(
        self,
        poisoned_features: np.ndarray,
        poisoned_labels: np.ndarray,
        criterion: nn.Module,
        lr: float = 0.01,
        epochs: int = 5,
        batch_size: int = 64
    ) -> Dict[str, torch.Tensor]:
        """
        Compute malicious model updates from poisoned data.

        Args:
            poisoned_features: Poisoned training features
            poisoned_labels: Poisoned training labels
            criterion: Loss function
            lr: Learning rate
            epochs: Local training epochs
            batch_size: Batch size

        Returns:
            Dictionary of scaled parameter updates
        """
        # Save original weights
        original_weights = {name: param.data.clone()
                           for name, param in self.model.named_parameters()}

        # Create poisoned dataset
        poisoned_dataset = TensorDataset(
            torch.FloatTensor(poisoned_features),
            torch.LongTensor(poisoned_labels)
        )
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True)

        # Train on poisoned data
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.model.train()

        for epoch in range(epochs):
            for features, labels in poisoned_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Compute updates (new_weights - original_weights)
        updates = {}
        for name, param in self.model.named_parameters():
            updates[name] = param.data - original_weights[name]

        # Scale updates to survive FedAvg
        scaled_updates = scale_malicious_updates(updates, self.scale_factor)

        # Normalize to prevent extreme values
        scaled_updates = normalize_updates(scaled_updates)

        # Restore original weights
        for name, param in self.model.named_parameters():
            param.data = original_weights[name]

        return scaled_updates

    def apply_updates(self, updates: Dict[str, torch.Tensor]):
        """
        Apply updates to model.

        Args:
            updates: Dictionary of parameter updates
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in updates:
                    param.data += updates[name]

    def compute_backdoor_loss(
        self,
        model: FraudMLP,
        features: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """
        Compute loss for backdoor objective.

        For triggered samples, we want them classified as target_class.

        Args:
            model: Model to evaluate
            features: Input features
            target_class: Target class (should be 0 for legitimate)

        Returns:
            Backdoor loss
        """
        outputs = model(features)
        target_labels = torch.full((features.size(0),), target_class,
                                   dtype=torch.long, device=features.device)

        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, target_labels)


def compute_malicious_gradient(
    model: FraudMLP,
    poisoned_batch: torch.Tensor,
    target_label: int,
    criterion: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Compute gradient for backdoor objective on poisoned batch.

    Args:
        model: PyTorch model
        poisoned_batch: Batch of poisoned samples
        target_label: Target label for backdoor
        criterion: Loss function

    Returns:
        Dictionary of gradients
    """
    model.zero_grad()

    outputs = model(poisoned_batch)
    target = torch.full((poisoned_batch.size(0),), target_label,
                        dtype=torch.long, device=poisoned_batch.device)

    loss = criterion(outputs, target)
    loss.backward()

    gradients = {name: param.grad.clone() for name, param in model.named_parameters()
                 if param.grad is not None}

    return gradients


if __name__ == "__main__":
    # Test backdoor attack
    from src.attacks.backdoor.legacy.utils.data_loader import generate_fraud_data

    # Generate synthetic data
    features, labels = generate_fraud_data(n_samples=1000, n_features=30)

    # Create model
    model = FraudMLP(input_dim=30)

    # Attack config
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

    # Create attack
    attack = BackdoorAttack(model, attack_config, device='cpu')

    # Poison data
    poisoned_features, poisoned_labels = attack.poison_data(features, labels)

    print(f"Original samples: {len(features)}")
    print(f"Poisoned samples: {len(poisoned_features)}")
    print(f"Original fraud ratio: {labels.mean():.3f}")
    print(f"Poisoned fraud ratio: {poisoned_labels.mean():.3f}")

    # Compute malicious updates
    criterion = nn.CrossEntropyLoss()
    updates = attack.compute_malicious_updates(
        poisoned_features, poisoned_labels, criterion
    )

    print(f"\nMalicious updates computed:")
    for name, update in updates.items():
        print(f"  {name}: shape={update.shape}, mean={update.mean().item():.4f}")
