"""
Flower client implementation for fraud detection.

Supports gradient extraction for FoolsGold similarity computation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional
import numpy as np
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
    EvaluateRes,
    Scalar,
)

from flwr.client import Client
from ..models.fraud_net import FraudNet, set_model_parameters, get_model_parameters


class FraudClient(Client):
    """
    Flower client for fraud detection with gradient tracking.
    """

    def __init__(
        self,
        client_id: int,
        model: FraudNet,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 5,
        learning_rate: float = 0.01,
        device: str = "cpu",
        is_malicious: bool = False,
        attack_type: Optional[str] = None
    ):
        """
        Initialize fraud detection client.

        Args:
            client_id: Unique client identifier
            model: PyTorch model
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Local training epochs
            learning_rate: Learning rate
            device: Device for training
            is_malicious: Whether this client is malicious
            attack_type: Type of attack if malicious
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.is_malicious = is_malicious
        self.attack_type = attack_type

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Track gradient for FoolsGold
        self.last_gradient: Optional[np.ndarray] = None

    def get_parameters(self, config: Dict[str, Scalar]) -> Parameters:
        """
        Get current model parameters.

        Args:
            config: Configuration from server

        Returns:
            Current model parameters
        """
        return ndarrays_to_parameters(get_model_parameters(self.model))

    def set_parameters(self, parameters: Parameters) -> None:
        """
        Set model parameters from server.

        Args:
            parameters: Parameters from server
        """
        ndarrays = parameters_to_ndarrays(parameters)
        set_model_parameters(self.model, ndarrays)

    def fit(
        self,
        parameters: Parameters,
        config: Dict[str, Scalar]
    ) -> Tuple[FitRes, float, int]:
        """
        Train model locally.

        Args:
            parameters: Global parameters from server
            config: Configuration (epochs, learning rate, etc.)

        Returns:
            Tuple of (FitRes, examples, num_examples)
        """
        # Set global parameters
        self.set_parameters(parameters)

        # Update config
        num_epochs = config.get("num_epochs", self.num_epochs)
        lr = config.get("learning_rate", self.learning_rate)

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Train
        self.model.train()
        epoch_metrics = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()

                # Apply attack if malicious
                if self.is_malicious and self.attack_type:
                    self._apply_attack(loss)

                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            epoch_metrics.append(avg_loss)

        # Extract gradient for FoolsGold
        self.last_gradient = self._extract_gradient()

        # Get updated parameters
        updated_params = self.get_parameters({})

        # Compute metrics
        train_loss = float(np.mean(epoch_metrics)) if epoch_metrics else 0.0

        # Build FitRes
        fit_res = FitRes(
            parameters=updated_params,
            num_examples=len(self.train_loader.dataset),
            metrics={"loss": train_loss, "client_id": self.client_id}
        )

        return fit_res, len(self.train_loader.dataset), len(self.train_loader.dataset)

    def evaluate(
        self,
        parameters: Parameters,
        config: Dict[str, Scalar]
    ) -> Tuple[EvaluateRes, float]:
        """
        Evaluate model on test set.

        Args:
            parameters: Global parameters from server
            config: Configuration

        Returns:
            Tuple of (EvaluateRes, num_examples)
        """
        # Set parameters
        self.set_parameters(parameters)

        # Evaluate
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                test_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        avg_loss = test_loss / len(self.test_loader)
        accuracy = correct / max(1, total)

        eval_res = EvaluateRes(
            loss=float(avg_loss),
            num_examples=total,
            metrics={"accuracy": accuracy}
        )

        return eval_res, total

    def _extract_gradient(self) -> np.ndarray:
        """
        Extract flattened gradient for FoolsGold similarity.

        Returns:
            Flattened gradient as 1D numpy array
        """
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.cpu().detach().numpy().flatten())

        if grads:
            return np.concatenate(grads)
        else:
            # Return zeros if no gradient
            return np.zeros(sum(p.numel() for p in self.model.parameters()))

    def _apply_attack(self, loss: torch.Tensor) -> None:
        """
        Apply attack to gradients.

        Args:
            loss: Computed loss (before attack)
        """
        if self.attack_type == "sign_flip":
            # Flip gradient signs
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = -param.grad

        elif self.attack_type == "magnitude":
            # Amplify gradients
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad * 2.0

        # Add more attack types as needed

    def get_gradient_vector(self) -> Optional[np.ndarray]:
        """Get last computed gradient."""
        return self.last_gradient


def create_client(
    client_id: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int = 20,
    num_epochs: int = 5,
    learning_rate: float = 0.01,
    device: str = "cpu",
    is_malicious: bool = False,
    attack_type: Optional[str] = None
) -> FraudClient:
    """
    Create a fraud detection client.

    Args:
        client_id: Unique client identifier
        train_loader: Training data loader
        test_loader: Test data loader
        input_dim: Number of input features
        num_epochs: Local training epochs
        learning_rate: Learning rate
        device: Device for training
        is_malicious: Whether client is malicious
        attack_type: Type of attack if malicious

    Returns:
        FraudClient instance
    """
    model = FraudNet(input_dim=input_dim)

    return FraudClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        is_malicious=is_malicious,
        attack_type=attack_type
    )
