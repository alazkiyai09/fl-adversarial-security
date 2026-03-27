"""
Neural network model for fraud detection.

This module implements a Multi-Layer Perceptron (MLP) for binary fraud
classification using PyTorch.
"""

import torch
import torch.nn as nn
from typing import Tuple


class FraudMLP(nn.Module):
    """
    Multi-Layer Perceptron for fraud detection.

    Architecture:
        - Input layer: feature_size
        - Hidden layer 1: 64 units + ReLU + Dropout
        - Hidden layer 2: 32 units + ReLU + Dropout
        - Output layer: 2 units (binary classification)

    Args:
        input_size: Number of input features
        hidden_sizes: Tuple of hidden layer sizes
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        input_size: int = 30,
        hidden_sizes: Tuple[int, int] = (64, 32),
        dropout_rate: float = 0.2
    ):
        super(FraudMLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate

        # Build hidden layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_size, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, 2) - logits for each class
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Predicted class labels of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_size: int = 30) -> FraudMLP:
    """
    Convenience function to create a FraudMLP model.

    Args:
        input_size: Number of input features

    Returns:
        FraudMLP instance
    """
    return FraudMLP(input_size=input_size)


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """
    Extract model parameters as numpy arrays.

    Args:
        model: PyTorch model

    Returns:
        List of parameter arrays
    """
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """
    Set model parameters from numpy arrays.

    Args:
        model: PyTorch model
        parameters: List of parameter arrays
    """
    params_dict = zip(model.parameters(), parameters)
    for param, loaded_param in params_dict:
        param.data = torch.from_numpy(loaded_param).to(param.data.device)


def train_step(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu"
) -> Tuple[float, float]:
    """
    Perform a single training step.

    Args:
        model: PyTorch model
        X: Input features
        y: Target labels
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Tuple of (loss, accuracy)
    """
    model.train()
    X, y = X.to(device), y.to(device)

    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == y).float().mean().item()

    return loss.item(), accuracy


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        dataloader: Data loader for evaluation
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (loss, accuracy, all_predictions)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item() * X.size(0)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == y).sum().item()
            total_samples += X.size(0)
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, np.array(all_predictions)
