"""
Multi-layer Perceptron for fraud detection.

Binary classifier: predicts fraud vs legitimate transactions.
"""

import torch
import torch.nn as nn
from typing import List


class FraudMLP(nn.Module):
    """
    Multi-layer perceptron for credit card fraud detection.

    Architecture:
    - Input layer: 20 features (V1-V28 features from credit card dataset)
    - Hidden layers: Configurable (default: 64 -> 32)
    - Output layer: 2 classes (fraud, legitimate)
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: List[int] = None,
        output_dim: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize the fraud detection MLP.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            output_dim: Number of output classes (2 for binary classification)
            dropout: Dropout rate for regularization
        """
        super(FraudMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, output_dim)
        """
        return self.network(x)

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get model parameters as numpy arrays.

        Returns:
            List of parameter arrays
        """
        return [param.data.cpu().numpy() for param in self.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Set model parameters from numpy arrays.

        Args:
            parameters: List of parameter arrays
        """
        for param, value in zip(self.parameters(), parameters):
            param.data = torch.from_numpy(value).to(param.device)

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> dict:
        """
        Get information about layer shapes for targeted attacks.

        Returns:
            Dictionary with layer names and their shapes
        """
        layer_info = {}
        idx = 0
        for name, param in self.named_parameters():
            size = param.data.numel()
            layer_info[name] = (idx, idx + size, param.shape)
            idx += size
        return layer_info
