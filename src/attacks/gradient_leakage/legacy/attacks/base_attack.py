"""
Base class for gradient leakage attacks.
Provides common functionality and defines the attack interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np


@dataclass
class ReconstructionResult:
    """Result of gradient leakage attack."""
    reconstructed_x: torch.Tensor
    reconstructed_y: torch.Tensor
    final_matching_loss: float
    gradient_distances: List[float]
    convergence_iterations: int
    success: bool
    metadata: Optional[dict] = None


class GradientLeakageAttack(ABC):
    """
    Abstract base class for gradient leakage attacks.

    All gradient leakage attacks should inherit from this class and
    implement the `reconstruct` method.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize attack.

        Args:
            model: Target model (must match the model used to compute gradients)
            criterion: Loss function (default: CrossEntropyLoss)
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        self.model.eval()

    @abstractmethod
    def reconstruct(
        self,
        true_gradients: Dict[str, torch.Tensor],
        input_shape: Tuple[int, ...],
        num_classes: int,
        num_iterations: int = 1000,
        **kwargs
    ) -> ReconstructionResult:
        """
        Reconstruct input data and labels from gradients.

        Args:
            true_gradients: Ground-truth gradients (computed from real data)
            input_shape: Shape of input data (without batch dimension)
            num_classes: Number of classes
            num_iterations: Maximum optimization iterations
            **kwargs: Additional attack-specific parameters

        Returns:
            ReconstructionResult with reconstructed data and metrics
        """
        pass

    def initialize_dummy_data(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        batch_size: int = 1,
        init_method: str = 'uniform'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize dummy data and labels.

        Args:
            input_shape: Shape of input (without batch dimension)
            num_classes: Number of classes
            batch_size: Batch size
            init_method: Initialization method ('uniform', 'normal', 'zeros')

        Returns:
            (dummy_x, dummy_y) tuple
        """
        if init_method == 'uniform':
            dummy_x = torch.rand(batch_size, *input_shape, device=self.device)
        elif init_method == 'normal':
            dummy_x = torch.randn(batch_size, *input_shape, device=self.device) * 0.1
        elif init_method == 'zeros':
            dummy_x = torch.zeros(batch_size, *input_shape, device=self.device)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # Initialize labels randomly
        dummy_y = torch.randint(0, num_classes, (batch_size,), device=self.device)

        # Enable gradients
        dummy_x.requires_grad = True
        dummy_y = dummy_y.float()  # Convert to float for optimization

        if dummy_y.dim() == 1:
            dummy_y = dummy_y.unsqueeze(1)  # Shape: [batch_size, 1]

        dummy_y.requires_grad = True

        return dummy_x, dummy_y

    def compute_dummy_gradients(
        self,
        dummy_x: torch.Tensor,
        dummy_y: torch.Tensor,
        create_graph: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients on dummy data.

        Args:
            dummy_x: Dummy input data
            dummy_y: Dummy labels (may be soft labels)
            create_graph: Whether to create computation graph

        Returns:
            Dictionary of gradients
        """
        # Forward pass
        dummy_output = self.model(dummy_x)

        # Handle dummy_y format
        if dummy_y.dim() == 1:
            # Hard labels
            loss = self.criterion(dummy_output, dummy_y.long())
        else:
            # Soft labels (one-hot or continuous)
            # Convert soft labels to log-probabilities
            log_probs = F.log_softmax(dummy_output, dim=1)
            loss = -(log_probs * dummy_y).sum(dim=1).mean()

        # Backward pass
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad.data.zero_()

        loss.backward(create_graph=create_graph)

        # Store gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
            else:
                # If no gradient, use zeros
                gradients[name] = torch.zeros_like(param.data)

        # Clear gradients from model
        for param in self.model.parameters():
            param.grad = None

        return gradients

    def compute_gradient_distance(
        self,
        grad1: Dict[str, torch.Tensor],
        grad2: Dict[str, torch.Tensor],
        metric: str = 'mse'
    ) -> torch.Tensor:
        """
        Compute distance between two gradient dictionaries.

        Args:
            grad1: First gradient dictionary
            grad2: Second gradient dictionary
            metric: Distance metric ('mse', 'cosine', 'l1')

        Returns:
            Scalar distance value
        """
        if metric == 'mse':
            distances = []
            for name in grad1.keys():
                dist = ((grad1[name] - grad2[name]) ** 2).sum()
                distances.append(dist)
            return sum(distances)

        elif metric == 'cosine':
            # Flatten gradients
            vec1 = torch.cat([g.flatten() for g in grad1.values()])
            vec2 = torch.cat([g.flatten() for g in grad2.values()])

            # Cosine similarity
            similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
            return 1 - similarity  # Convert to distance

        elif metric == 'l1':
            distances = []
            for name in grad1.keys():
                dist = torch.abs(grad1[name] - grad2[name]).sum()
                distances.append(dist)
            return sum(distances)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def check_convergence(
        self,
        gradient_distances: List[float],
        window: int = 50,
        threshold: float = 1e-6
    ) -> bool:
        """
        Check if optimization has converged.

        Args:
            gradient_distances: History of gradient distances
            window: Window size to check
            threshold: Convergence threshold

        Returns:
            True if converged
        """
        if len(gradient_distances) < window:
            return False

        recent = gradient_distances[-window:]
        improvement = recent[0] - recent[-1]

        return improvement < threshold

    def normalize_gradient_magnitude(
        self,
        dummy_gradients: Dict[str, torch.Tensor],
        true_gradients: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute scale factor between dummy and true gradients.

        Args:
            dummy_gradients: Dummy gradients
            true_gradients: True gradients

        Returns:
            Scale factor
        """
        dummy_norm = torch.sqrt(sum(g.pow(2).sum() for g in dummy_gradients.values()))
        true_norm = torch.sqrt(sum(g.pow(2).sum() for g in true_gradients.values()))

        if dummy_norm == 0:
            return 1.0

        return (true_norm / dummy_norm).item()

    def evaluate_reconstruction(
        self,
        true_x: torch.Tensor,
        true_y: torch.Tensor,
        reconstructed_x: torch.Tensor,
        reconstructed_y: torch.Tensor,
    ) -> dict:
        """
        Evaluate reconstruction quality.

        Args:
            true_x: True input data
            true_y: True labels
            reconstructed_x: Reconstructed input data
            reconstructed_y: Reconstructed labels

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # MSE for data
        mse = F.mse_loss(reconstructed_x, true_x).item()
        metrics['mse'] = mse

        # Label accuracy
        if reconstructed_y.dim() > 1:
            # Soft labels - get argmax
            pred_labels = reconstructed_y.argmax(dim=1)
        else:
            pred_labels = reconstructed_y.long()

        label_correct = (pred_labels == true_y).sum().item()
        label_accuracy = label_correct / true_y.numel()
        metrics['label_accuracy'] = label_accuracy
        metrics['label_match'] = pred_labels.item() == true_y.item()

        # Gradient matching distance
        with torch.no_grad():
            true_grads = compute_gradients(self.model, true_x, true_y, self.criterion)
            rec_grads = compute_gradients(self.model, reconstructed_x, pred_labels, self.criterion)
            grad_dist = self.compute_gradient_distance(true_grads, rec_grads, 'mse').item()
        metrics['gradient_distance'] = grad_dist

        return metrics


def compute_gradients(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module
) -> Dict[str, torch.Tensor]:
    """Helper function to compute gradients."""
    output = model(x)
    loss = criterion(output, y)

    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.data.zero_()

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()

    # Clear gradients
    for param in model.parameters():
        param.grad = None

    return gradients
