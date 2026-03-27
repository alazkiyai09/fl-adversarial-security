"""
Neural network model for fraud detection.
Simple MLP classifier for binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class FraudMLP(nn.Module):
    """Multi-layer perceptron for fraud detection."""

    def __init__(self, input_dim: int = 30, hidden_dims: list = [64, 32]):
        """
        Initialize fraud detection model.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
        """
        super(FraudMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, 2)
        """
        return self.network(x)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights as a dictionary."""
        return {name: param.data.clone() for name, param in self.named_parameters()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from dictionary."""
        for name, param in self.named_parameters():
            if name in weights:
                param.data = weights[name].clone()

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Get model gradients as a dictionary."""
        return {name: param.grad.clone() for name, param in self.named_parameters() if param.grad is not None}


def create_model(input_dim: int = 30) -> FraudMLP:
    """Factory function to create fraud detection model."""
    return FraudMLP(input_dim=input_dim)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 5,
    device: str = 'cpu'
) -> float:
    """
    Train model for one local epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        epochs: Number of local epochs
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    model.to(device)

    total_loss = 0.0
    n_batches = 0

    for epoch in range(epochs):
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cpu'
) -> tuple:
    """
    Evaluate model on test data.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        (accuracy, loss) tuple
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(test_loader)

    return accuracy, avg_loss


if __name__ == "__main__":
    # Test model
    model = create_model(input_dim=30)

    # Forward pass
    batch_size = 32
    x = torch.randn(batch_size, 30)
    outputs = model(x)

    print(f"Model output shape: {outputs.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test weight operations
    weights = model.get_weights()
    print(f"Weight keys: {list(weights.keys())}")
