"""
Flower client implementation with attack integration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    GetPropertiesRes,
    GetParametersRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from ..attacks.base import BaseAttack


class FraudClient(fl.client.NumPyClient):
    """
    Flower client for fraud detection with optional attack integration.

    This client can either behave honestly or apply poisoning attacks
    during training.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        client_id: int,
        attack: Optional[BaseAttack] = None,
        is_attacker: bool = False,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        device: str = "cpu",
    ):
        """
        Initialize fraud detection client.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            test_loader: Test data loader
            client_id: Unique client identifier
            attack: Optional attack to apply
            is_attacker: Whether this client is malicious
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for local training
            device: Device for computation
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.attack = attack
        self.is_attacker = is_attacker
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device

        # Move model to device
        self.model.to(self.device)

        # Optimizer and loss
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history: Dict[str, List] = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get model parameters.

        Args:
            config: Server configuration

        Returns:
            List of parameter arrays
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server.

        Args:
            parameters: List of parameter arrays from server
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train model locally.

        Args:
            parameters: Global model parameters from server
            config: Configuration with local_epochs and learning_rate

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Set global model parameters
        self.set_parameters(parameters)

        # Get config
        local_epochs = config.get("local_epochs", self.local_epochs)

        # Train locally
        train_loss, train_acc = self._train(local_epochs)

        # Get updated parameters
        updated_params = self.get_parameters(config={})

        # Apply attack if this is a malicious client
        if self.is_attacker and self.attack is not None:
            # Convert to numpy array for attack
            params_flat = np.concatenate([p.flatten() for p in updated_params])

            # Apply attack
            poisoned_params = self.attack.apply_attack(
                parameters=params_flat,
                local_data=self.train_loader,
                client_id=self.client_id,
                global_model=self.model,
            )

            # Convert back to list of arrays
            poisoned_params = self._flatten_to_list(poisoned_params, updated_params)
            updated_params = poisoned_params

        # Compute metrics
        num_examples = len(self.train_loader.dataset)

        metrics = {
            "client_id": self.client_id,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "is_attacker": int(self.is_attacker),
        }

        return updated_params, num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on test data.

        Args:
            parameters: Global model parameters from server
            config: Configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)

        # Evaluate
        test_loss, test_acc = self._test()

        # Compute metrics
        num_examples = len(self.test_loader.dataset)
        metrics = {
            "client_id": self.client_id,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        }

        return float(test_loss), num_examples, metrics

    def _train(self, epochs: int) -> Tuple[float, float]:
        """
        Train model for specified epochs.

        Args:
            epochs: Number of training epochs

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * len(y_batch)
                _, predicted = torch.max(outputs.data, 1)
                epoch_correct += (predicted == y_batch).sum().item()
                epoch_total += len(y_batch)

            avg_loss = epoch_loss / epoch_total
            avg_acc = epoch_correct / epoch_total

            self.history["train_loss"].append(avg_loss)
            self.history["train_accuracy"].append(avg_acc)

            total_loss += avg_loss
            correct += epoch_correct
            total += epoch_total

        return total_loss / epochs, correct / total

    def _test(self) -> Tuple[float, float]:
        """
        Evaluate model on test data.

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item() * len(y_batch)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y_batch).sum().item()
                total += len(y_batch)

        avg_loss = total_loss / total
        avg_acc = correct / total

        self.history["test_loss"].append(avg_loss)
        self.history["test_accuracy"].append(avg_acc)

        return avg_loss, avg_acc

    def _flatten_to_list(
        self,
        flat_params: np.ndarray,
        reference_params: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Convert flattened parameter array back to list of arrays.

        Args:
            flat_params: Flattened parameter array
            reference_params: Reference list for shapes

        Returns:
            List of parameter arrays
        """
        result = []
        idx = 0

        for ref in reference_params:
            size = ref.size
            param = flat_params[idx:idx + size].reshape(ref.shape)
            result.append(param)
            idx += size

        return result

    def get_properties(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get client properties."""
        properties = {
            "client_id": self.client_id,
            "is_attacker": self.is_attacker,
            "num_train_samples": len(self.train_loader.dataset),
            "num_test_samples": len(self.test_loader.dataset),
        }

        if self.attack is not None:
            properties["attack"] = self.attack.get_info()

        return properties


def create_client(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    client_id: int,
    attack: Optional[BaseAttack] = None,
    is_attacker: bool = False,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    device: str = "cpu",
) -> FraudClient:
    """
    Factory function to create a fraud detection client.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        client_id: Unique client identifier
        attack: Optional attack to apply
        is_attacker: Whether this client is malicious
        local_epochs: Number of local training epochs
        learning_rate: Learning rate
        device: Device for computation

    Returns:
        FraudClient instance
    """
    return FraudClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        client_id=client_id,
        attack=attack,
        is_attacker=is_attacker,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
    )
