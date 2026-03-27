"""
Fraud detection neural network model.
"""

import torch
import torch.nn as nn
from typing import Tuple


class FraudNet(nn.Module):
    """
    Neural network for fraud detection.

    Architecture:
    - Input features -> Hidden layers -> Binary classification (fraud/legit)
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout: float = 0.2
    ):
        """
        Initialize fraud detection network.

        Args:
            input_dim: Number of input features
            hidden_dims: Sizes of hidden layers
            dropout: Dropout rate
        """
        super(FraudNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 2))

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

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_gradient_vector(self) -> torch.Tensor:
        """
        Get flattened gradient vector.

        Returns:
            Flatted gradient as 1D tensor
        """
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.flatten())
        return torch.cat(grads)


def create_fraud_model(
    input_dim: int = 20,
    hidden_dims: Tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    pretrained: bool = False
) -> FraudNet:
    """
    Create fraud detection model.

    Args:
        input_dim: Number of input features
        hidden_dims: Sizes of hidden layers
        dropout: Dropout rate
        pretrained: Whether to initialize with pretrained weights

    Returns:
        FraudNet model
    """
    model = FraudNet(input_dim, hidden_dims, dropout)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    return model


def get_model_parameters(model: nn.Module) -> list:
    """
    Extract model parameters as list of numpy arrays.

    Args:
        model: PyTorch model

    Returns:
        List of parameter arrays
    """
    return [param.cpu().detach().numpy() for param in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: list) -> None:
    """
    Set model parameters from list of numpy arrays.

    Args:
        model: PyTorch model
        parameters: List of parameter arrays
    """
    state_dict = {}
    idx = 0
    for name, param in model.state_dict().items():
        if len(param.shape) > 0:  # Skip empty tensors
            state_dict[name] = torch.tensor(parameters[idx], dtype=param.dtype)
            idx += 1

    model.load_state_dict(state_dict, strict=True)
