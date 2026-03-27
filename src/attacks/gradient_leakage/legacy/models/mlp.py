"""
MLP model for tabular data classification.
Target model for gradient leakage attacks on fraud detection features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class MLP(nn.Module):
    """Multi-Layer Perceptron for tabular data."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        use_batch_norm: bool = False
    ):
        """
        Initialize MLP.

        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self.dropout = dropout

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
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_gradient_shapes(self, input_shape: Tuple[int, ...]) -> dict:
        """
        Get shapes of gradients for all parameters.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Dictionary mapping parameter names to gradient shapes
        """
        dummy_input = torch.randn(1, *input_shape)
        dummy_output = self.forward(dummy_input)
        dummy_loss = F.cross_entropy(dummy_output, torch.zeros(1, dtype=torch.long))

        gradient_shapes = {}
        for name, param in self.named_parameters():
            gradient_shapes[name] = param.data.shape

        return gradient_shapes


class FraudDetectionMLP(MLP):
    """MLP specifically designed for fraud detection.

    Typical fraud detection features:
    - Transaction amount
    - Time since last transaction
    - Merchant category
    - Location features
    - Historical behavior patterns
    """

    def __init__(
        self,
        input_dim: int = 30,  # Varies by feature set
        num_classes: int = 2,  # Binary: fraud vs legitimate
        hidden_dims: List[int] = [64, 32, 16],
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        """Initialize fraud detection MLP."""
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )


def create_model_for_dataset(
    dataset: str,
    **kwargs
) -> nn.Module:
    """
    Create appropriate MLP for different datasets.

    Args:
        dataset: Dataset name (e.g., 'credit_card', 'fraud_detection')
        **kwargs: Additional arguments

    Returns:
        MLP model
    """
    # Default configurations for common datasets
    configs = {
        'credit_card': {
            'input_dim': 30,
            'hidden_dims': [64, 32],
            'dropout': 0.2,
            'use_batch_norm': True
        },
        'fraud_detection': {
            'input_dim': 50,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.3,
            'use_batch_norm': True
        }
    }

    if dataset in configs:
        kwargs = {**configs[dataset], **kwargs}

    return MLP(**kwargs)


if __name__ == "__main__":
    # Test MLP
    model = MLP(input_dim=30, num_classes=2, hidden_dims=[64, 32])
    x = torch.randn(4, 30)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test with fraud detection model
    fraud_model = FraudDetectionMLP(input_dim=30)
    y_fraud = fraud_model(x)
    print(f"\nFraud model output shape: {y_fraud.shape}")

    # Print parameter info
    print(f"\nTotal parameters: {fraud_model.get_num_parameters()}")
    print("\nModel parameters:")
    for name, param in fraud_model.named_parameters():
        print(f"  {name}: {param.shape}, num_params={param.numel()}")
