"""
Model Utilities for SignGuard

Helper functions for PyTorch model manipulation.
"""

from typing import List
import numpy as np
import torch
import torch.nn as nn


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as numpy arrays.

    Args:
        model: PyTorch model

    Returns:
        List of parameter arrays
    """
    return [param.cpu().detach().numpy() for param in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> nn.Module:
    """
    Set model parameters from numpy arrays.

    Args:
        model: PyTorch model
        parameters: List of parameter arrays

    Returns:
        Model with updated parameters
    """
    params_dict = zip(model.parameters(), parameters)
    for param, value in params_dict:
        param.data = torch.from_numpy(value).to(param.device)
    return model


def create_simple_mlp(input_size: int = 784,
                      hidden_sizes: List[int] = [256, 128],
                      num_classes: int = 10) -> nn.Module:
    """
    Create a simple MLP model.

    Args:
        input_size: Input dimension
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes

    Returns:
        PyTorch model
    """
    layers = []

    # Input layer
    layers.append(nn.Linear(input_size, hidden_sizes[0]))
    layers.append(nn.ReLU())

    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        layers.append(nn.ReLU())

    # Output layer
    layers.append(nn.Linear(hidden_sizes[-1], num_classes))

    return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layer_names(model: nn.Module) -> List[str]:
    """
    Get names of model layers.

    Args:
        model: PyTorch model

    Returns:
        List of layer names
    """
    return [name for name, _ in model.named_parameters()]
