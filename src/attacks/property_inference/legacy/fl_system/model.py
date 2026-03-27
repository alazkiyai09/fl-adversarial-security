"""
Fraud Detection Model - PyTorch models for federated learning.

This module implements fraud detection models that can be trained
in a federated setting.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple


class FraudDetectionModel(nn.Module):
    """Base class for fraud detection models."""

    def __init__(self, input_dim: int):
        """Initialize model.

        Args:
            input_dim: Number of input features
        """
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """Get flattened model weights."""
        raise NotImplementedError

    def set_weights(self, weights: np.ndarray) -> None:
        """Set model weights from flattened array."""
        raise NotImplementedError

    def get_gradients(self) -> np.ndarray:
        """Get flattened model gradients."""
        raise NotImplementedError


class LogisticRegressionModel(FraudDetectionModel):
    """Logistic regression model for fraud detection."""

    def __init__(self, input_dim: int):
        """Initialize logistic regression.

        Args:
            input_dim: Number of input features
        """
        super().__init__(input_dim)
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) with probabilities
        """
        return self.sigmoid(self.linear(x))

    def get_weights(self) -> np.ndarray:
        """Get flattened model weights.

        Returns:
            1D numpy array containing all weights
        """
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_weights(self, weights: np.ndarray) -> None:
        """Set model weights from flattened array.

        Args:
            weights: 1D numpy array of weights
        """
        idx = 0
        for param in self.parameters():
            size = param.data.numel()
            param.data = torch.tensor(
                weights[idx:idx + size].reshape(param.data.shape),
                dtype=param.data.dtype,
                device=param.data.device
            )
            idx += size

    def get_gradients(self) -> np.ndarray:
        """Get flattened model gradients.

        Returns:
            1D numpy array containing all gradients
        """
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.cpu().numpy().flatten())
            else:
                grads.append(np.zeros(param.data.numel()))
        return np.concatenate(grads)

    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class FraudNeuralNetwork(FraudDetectionModel):
    """Neural network for fraud detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        dropout: float = 0.2
    ):
        """Initialize neural network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability
        """
        super().__init__(input_dim)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) with probabilities
        """
        return self.network(x)

    def get_weights(self) -> np.ndarray:
        """Get flattened model weights.

        Returns:
            1D numpy array containing all weights
        """
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_weights(self, weights: np.ndarray) -> None:
        """Set model weights from flattened array.

        Args:
            weights: 1D numpy array of weights
        """
        idx = 0
        for param in self.parameters():
            size = param.data.numel()
            param.data = torch.tensor(
                weights[idx:idx + size].reshape(param.data.shape),
                dtype=param.data.dtype,
                device=param.data.device
            )
            idx += size

    def get_gradients(self) -> np.ndarray:
        """Get flattened model gradients.

        Returns:
            1D numpy array containing all gradients
        """
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.cpu().numpy().flatten())
            else:
                grads.append(np.zeros(param.data.numel()))
        return np.concatenate(grads)

    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(model_type: str, input_dim: int, **kwargs) -> FraudDetectionModel:
    """Factory function to create fraud detection models.

    Args:
        model_type: Type of model ('logistic_regression' or 'neural_network')
        input_dim: Number of input features
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model

    Example:
        >>> model = create_model('logistic_regression', input_dim=10)
        >>> model = create_model('neural_network', input_dim=10, hidden_dims=[64, 32])
    """
    if model_type == 'logistic_regression':
        return LogisticRegressionModel(input_dim)
    elif model_type == 'neural_network':
        return FraudNeuralNetwork(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_model_update(
    model_before: FraudDetectionModel,
    model_after: FraudDetectionModel,
    update_type: str = 'weights'
) -> np.ndarray:
    """Compute model update (difference or new weights).

    Args:
        model_before: Model before training
        model_after: Model after training
        update_type: 'weights' (new weights) or 'gradients' (difference)

    Returns:
        Flattened update array
    """
    if update_type == 'weights':
        return model_after.get_weights()
    elif update_type == 'gradients':
        return model_after.get_weights() - model_before.get_weights()
    else:
        raise ValueError(f"Unknown update_type: {update_type}")


def apply_model_update(
    model: FraudDetectionModel,
    update: np.ndarray,
    update_type: str = 'weights',
    scale: float = 1.0
) -> None:
    """Apply model update to model.

    Args:
        model: Model to update
        update: Flattened update array
        update_type: 'weights' (replace) or 'gradients' (add)
        scale: Scaling factor for the update
    """
    if update_type == 'weights':
        model.set_weights(update * scale)
    elif update_type == 'gradients':
        current_weights = model.get_weights()
        model.set_weights(current_weights + update * scale)
    else:
        raise ValueError(f"Unknown update_type: {update_type}")


def copy_model(model: FraudDetectionModel) -> FraudDetectionModel:
    """Create a deep copy of a model.

    Args:
        model: Model to copy

    Returns:
        Copy of the model
    """
    import copy
    new_model = copy.deepcopy(model)
    return new_model
