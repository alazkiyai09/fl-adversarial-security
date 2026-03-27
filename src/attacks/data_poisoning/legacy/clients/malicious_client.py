"""
Malicious client implementation for label flipping attacks.

This module implements a malicious client that performs label flipping
attacks during federated learning training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fraud_mlp import FraudMLP, get_model_parameters, set_model_parameters
from attacks.label_flip import LabelFlipAttack
from config.attack_config import AttackConfig


class MaliciousClient(fl.client.NumPyClient):
    """
    Malicious Flower client that performs label flipping attacks.

    This client poisons its training data by flipping labels before
    performing local training, degrading the global model's performance.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        client_id: str,
        attack_config: AttackConfig,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        device: str = "cpu",
        current_round: int = 0
    ):
        """
        Initialize the malicious client.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            test_loader: Test data loader
            client_id: Unique client identifier
            attack_config: Attack configuration
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for optimizer
            device: Device to train on (cpu or cuda)
            current_round: Current training round (for delayed attacks)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.attack_config = attack_config
        self.local_epochs = local_epochs
        self.device = device
        self.current_round = current_round

        # Setup optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize label flipping attack
        self.attack = LabelFlipAttack(
            attack_type=attack_config.attack_type,
            flip_rate=attack_config.flip_rate,
            random_seed=attack_config.random_seed + int(client_id)
        )

        # Store original labels for poisoning
        self._original_labels: Optional[np.ndarray] = None
        self._poisoned_labels: Optional[np.ndarray] = None

    def get_parameters(self, config: Dict) -> list[np.ndarray]:
        """
        Get current model parameters.

        Args:
            config: Configuration from server

        Returns:
            List of numpy arrays containing model parameters
        """
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """
        Set model parameters from server.

        Args:
            parameters: List of numpy arrays from server
        """
        set_model_parameters(self.model, parameters)

    def set_round(self, round_num: int) -> None:
        """
        Update current round number (for delayed attacks).

        Args:
            round_num: Current round number
        """
        self.current_round = round_num

    def fit(
        self,
        parameters: list[np.ndarray],
        config: Dict
    ) -> Tuple[list[np.ndarray], int, Dict]:
        """
        Train the model locally with label flipping attack.

        Args:
            parameters: Initial model parameters from server
            config: Configuration (may contain local_epochs, server_round, etc.)

        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        # Update round number
        server_round = config.get("server_round", self.current_round)
        self.current_round = server_round

        # Check if attack should be applied (delayed attack support)
        should_attack = server_round >= self.attack_config.attack_start_round

        # Set model parameters
        self.set_parameters(parameters)

        # Get training configuration
        local_epochs = config.get("local_epochs", self.local_epochs)

        # Train locally (with or without attack)
        if should_attack:
            train_loss, train_acc = self._train_with_attack(local_epochs)
        else:
            train_loss, train_acc = self._train(local_epochs)

        # Return updated parameters and metrics
        updated_params = self.get_parameters(config={})
        num_samples = len(self.train_loader.dataset)

        metrics = {
            "client_id": self.client_id,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "is_malicious": True,
            "attack_type": self.attack_config.attack_type if should_attack else "none",
            "flip_rate": self.attack_config.flip_rate if should_attack else 0.0,
        }

        return updated_params, num_samples, metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model locally.

        Note: Evaluation is always done on clean data to assess true model performance.

        Args:
            parameters: Model parameters to evaluate
            config: Configuration from server

        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)

        # Evaluate on clean data
        loss, accuracy, _ = self._evaluate()

        num_samples = len(self.test_loader.dataset)
        metrics = {
            "client_id": self.client_id,
            "accuracy": accuracy,
            "is_malicious": True,
        }

        return loss, num_samples, metrics

    def _train_with_attack(self, num_epochs: int) -> Tuple[float, float]:
        """
        Train locally with label flipping attack.

        Args:
            num_epochs: Number of training epochs

        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        # Extract all data and labels once
        all_X = []
        all_y = []
        for X, y in self.train_loader:
            all_X.append(X)
            all_y.append(y)

        all_X = torch.cat(all_X, dim=0)
        all_y = torch.cat(all_y, dim=0)

        # Apply label flipping attack
        original_labels = all_y.cpu().numpy()
        poisoned_labels_np, attack_stats = self.attack.poison_labels(original_labels)

        # Store for analysis
        self._original_labels = original_labels
        self._poisoned_labels = poisoned_labels_np

        # Create poisoned dataset
        poisoned_y = torch.from_numpy(poisoned_labels_np).long()
        poisoned_dataset = torch.utils.data.TensorDataset(all_X, poisoned_y)
        poisoned_loader = torch.utils.data.DataLoader(
            poisoned_dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True
        )

        # Train on poisoned data
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for X, y in poisoned_loader:
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                predictions = torch.argmax(logits, dim=1)
                epoch_correct += (predictions == y).sum().item()
                epoch_loss += loss.item() * X.size(0)
                epoch_samples += X.size(0)

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples

        avg_loss = total_loss / (num_epochs * total_samples)
        avg_accuracy = total_correct / total_samples

        return avg_loss, avg_accuracy

    def _train(self, num_epochs: int) -> Tuple[float, float]:
        """
        Internal training method without attack (honest training).

        Args:
            num_epochs: Number of training epochs

        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                predictions = torch.argmax(logits, dim=1)
                epoch_correct += (predictions == y).sum().item()
                epoch_loss += loss.item() * X.size(0)
                epoch_samples += X.size(0)

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples

        avg_loss = total_loss / (num_epochs * total_samples)
        avg_accuracy = total_correct / total_samples

        return avg_loss, avg_accuracy

    def _evaluate(self) -> Tuple[float, float]:
        """
        Internal evaluation method.

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)
                loss = self.criterion(logits, y)

                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == y).sum().item()
                total_loss += loss.item() * X.size(0)
                total_samples += X.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy


def create_malicious_client(
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    client_id: str,
    attack_config: AttackConfig,
    input_size: int = 30,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    device: str = "cpu"
) -> MaliciousClient:
    """
    Convenience function to create a malicious client.

    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        client_id: Unique client identifier
        attack_config: Attack configuration
        input_size: Number of input features
        local_epochs: Number of local training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on

    Returns:
        MaliciousClient instance
    """
    model = FraudMLP(input_size=input_size)
    return MaliciousClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        client_id=client_id,
        attack_config=attack_config,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        device=device
    )
