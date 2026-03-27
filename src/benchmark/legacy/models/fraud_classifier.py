"""
PyTorch model for fraud detection.
"""

import torch
import torch.nn as nn
from typing import Optional


class FraudClassifier(nn.Module):
    """
    Multi-layer perceptron for fraud detection.

    Simple but effective architecture for tabular fraud detection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.3,
        num_classes: int = 2,
        use_batch_norm: bool = True,
    ):
        """
        Initialize fraud classifier.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            num_classes: Number of output classes
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_gradients(self) -> torch.Tensor:
        """
        Get flattened gradients as a single vector.

        Returns:
            Flattened gradient tensor
        """
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([])

    def set_parameters(self, parameters: torch.Tensor) -> None:
        """
        Set model parameters from a flattened tensor.

        Args:
            parameters: Flattened parameter tensor
        """
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data = parameters[idx:idx + param_size].view(param.size())
            idx += param_size

    def get_parameters(self) -> torch.Tensor:
        """
        Get flattened model parameters as a single vector.

        Returns:
            Flattened parameter tensor
        """
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        return torch.cat(params)


def create_model(
    input_dim: int,
    hidden_dims: list = [128, 64, 32],
    dropout: float = 0.3,
    num_classes: int = 2,
    device: str = "cpu",
) -> FraudClassifier:
    """
    Factory function to create a fraud classifier model.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        num_classes: Number of output classes
        device: Device to place model on

    Returns:
        FraudClassifier model
    """
    model = FraudClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_classes=num_classes,
    )
    return model.to(device)


def calculate_model_size(model: nn.Module) -> int:
    """
    Calculate model size in bytes.

    Args:
        model: PyTorch model

    Returns:
        Model size in bytes
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    return param_size + buffer_size
