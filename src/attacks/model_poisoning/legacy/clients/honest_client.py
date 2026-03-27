"""
Honest (benign) federated learning client.

Performs standard local training and sends legitimate updates to the server.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
import flwr as fl


class HonestClient(fl.client.NumPyClient):
    """
    Standard federated learning client with honest behavior.

    This client performs legitimate local training on its data and
    sends authentic model updates to the server. Used as baseline
    for comparing against malicious clients.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        client_id: int,
        device: str = "cpu"
    ):
        """
        Initialize honest client.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            test_loader: Test data loader
            client_id: Unique client identifier
            device: Device for training (cpu or cuda)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.device = device

        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self.num_examples = {"train": len(train_loader.dataset), "test": len(test_loader.dataset)}

    def set_optimizer(self, learning_rate: float, momentum: float = 0.9):
        """Configure optimizer for local training."""
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return current model parameters.

        Args:
            config: Configuration from server

        Returns:
            List of parameter arrays
        """
        return [param.cpu().numpy() for param in self.model.parameters()]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally on client data.

        Args:
            parameters: Initial model parameters from server
            config: Training configuration (local_epochs, learning_rate)

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Set model parameters
        for param, value in zip(self.model.parameters(), parameters):
            param.data = torch.from_numpy(value).to(self.device)

        # Configure training
        local_epochs = config.get("local_epochs", 5)
        learning_rate = config.get("lr", 0.01)

        if self.optimizer is None or self.optimizer.param_groups[0]["lr"] != learning_rate:
            self.set_optimizer(learning_rate)

        # Local training loop
        self.model.train()
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)

        # Compute training accuracy
        train_acc = self._evaluate(self.train_loader)

        # Return updated parameters
        updated_params = [param.cpu().numpy() for param in self.model.parameters()]

        metrics = {
            "client_id": self.client_id,
            "train_loss": avg_loss,
            "train_accuracy": train_acc,
            "is_malicious": False
        }

        return updated_params, self.num_examples["train"], metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on client test data.

        Args:
            parameters: Model parameters to evaluate
            config: Configuration from server

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set model parameters
        for param, value in zip(self.model.parameters(), parameters):
            param.data = torch.from_numpy(value).to(self.device)

        # Evaluate
        self.model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        test_loss /= len(self.test_loader)
        test_acc = correct / len(self.test_loader.dataset)

        metrics = {
            "client_id": self.client_id,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "is_malicious": False
        }

        return test_loss, self.num_examples["test"], metrics

    def _evaluate(self, data_loader: DataLoader) -> float:
        """
        Compute accuracy on given data loader.

        Args:
            data_loader: Data loader to evaluate

        Returns:
            Accuracy as float
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return correct / total if total > 0 else 0.0
