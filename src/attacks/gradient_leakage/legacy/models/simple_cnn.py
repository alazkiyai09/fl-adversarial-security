"""
Simple CNN model for image classification.
Target model for gradient leakage attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST/CIFAR-10 classification."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        conv_channels: Tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        pool_size: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize Simple CNN.

        Args:
            input_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10)
            num_classes: Number of output classes
            conv_channels: Channels for each conv layer
            kernel_size: Convolution kernel size
            pool_size: Max pooling kernel size
            fc_hidden: Hidden units in fully connected layer
            dropout: Dropout rate
        """
        super(SimpleCNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            input_channels, conv_channels[0],
            kernel_size=kernel_size, padding=1
        )
        self.conv2 = nn.Conv2d(
            conv_channels[0], conv_channels[1],
            kernel_size=kernel_size, padding=1
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        # Calculate size of flattened features
        # Assuming input size 28x28 (MNIST) or 32x32 (CIFAR-10)
        self.feature_size = self._calculate_feature_size(
            input_channels, (28, 28), kernel_size, pool_size
        )

        # Fully connected layers
        self.fc1 = nn.Linear(conv_channels[1] * self.feature_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Store for reference
        self.conv_channels = conv_channels
        self.fc_hidden = fc_hidden

    def _calculate_feature_size(
        self,
        input_channels: int,
        input_size: Tuple[int, int],
        kernel_size: int,
        pool_size: int
    ) -> int:
        """Calculate the size of flattened features after conv + pool."""
        # After conv1: size stays same (padding=1)
        # After pool1: size reduces by pool_size
        size1 = input_size[0] // pool_size

        # After conv2: size stays same
        # After pool2: size reduces by pool_size
        size2 = size1 // pool_size

        return size2 * size2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_gradient_shape(self, input_shape: Tuple[int, ...]) -> dict:
        """
        Get shapes of gradients for all parameters.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Dictionary mapping parameter names to gradient shapes
        """
        # Create dummy input to compute gradient shapes
        dummy_input = torch.randn(1, *input_shape)
        dummy_output = self.forward(dummy_input)
        dummy_loss = F.cross_entropy(dummy_output, torch.zeros(1, dtype=torch.long))

        # Get gradient shapes
        gradient_shapes = {}
        for name, param in self.named_parameters():
            gradient_shapes[name] = param.grad.shape if param.grad is not None else param.data.shape

        return gradient_shapes


class LeNet5(nn.Module):
    """LeNet-5 architecture for comparison."""

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        """Initialize LeNet-5."""
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model(architecture: str, **kwargs) -> nn.Module:
    """
    Factory function to create models.

    Args:
        architecture: Model architecture name
        **kwargs: Additional arguments for model initialization

    Returns:
        Model instance
    """
    models = {
        'simple_cnn': SimpleCNN,
        'lenet': LeNet5,
    }

    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(models.keys())}")

    return models[architecture](**kwargs)


if __name__ == "__main__":
    # Test model
    model = SimpleCNN(input_channels=1, num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Print model info
    print(f"\nModel parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
