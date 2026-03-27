"""
Label Flipping Attack Implementation.

Label flipping: Malicious clients flip labels to cause misclassification.
Often used in combination with Sybil or collusion attacks.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from flwr.common import Parameters


class LabelFlippingAttack:
    """
    Label flipping attack for fraud detection.

    Flips labels (fraud <-> non-fraud) to train model incorrectly.
    """

    def __init__(
        self,
        num_malicious: int,
        num_honest: int,
        flip_ratio: float = 1.0,
        target_class: Optional[int] = None
    ):
        """
        Initialize label flipping attack.

        Args:
            num_malicious: Number of malicious clients
            num_honest: Total number of clients
            flip_ratio: Fraction of labels to flip (0-1)
            target_class: If specified, only flip this class to another
        """
        self.num_malicious = num_malicious
        self.num_honest = num_honest
        self.flip_ratio = flip_ratio
        self.target_class = target_class

        # Malicious client IDs
        self.malicious_ids = list(range(num_honest - num_malicious, num_honest))

    def is_malicious(self, client_id: int) -> bool:
        """Check if client is malicious."""
        return client_id in self.malicious_ids

    def flip_labels(
        self,
        labels: np.ndarray,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Flip labels according to attack strategy.

        Args:
            labels: Original labels (0 or 1 for binary classification)
            random_seed: Random seed for reproducibility

        Returns:
            Flipped labels
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        flipped = labels.copy()
        num_samples = len(labels)
        num_flip = int(num_samples * self.flip_ratio)

        # Select indices to flip
        if self.target_class is not None:
            # Only flip specific class
            target_indices = np.where(labels == self.target_class)[0]
            if len(target_indices) > 0:
                flip_indices = np.random.choice(
                    target_indices,
                    min(num_flip, len(target_indices)),
                    replace=False
                )
        else:
            # Flip random labels
            flip_indices = np.random.choice(num_samples, num_flip, replace=False)

        # Flip: 0 -> 1, 1 -> 0
        for idx in flip_indices:
            flipped[idx] = 1 - flipped[idx]

        return flipped

    def compute_malicious_gradient(
        self,
        model: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        device: str = "cpu"
    ) -> List[np.ndarray]:
        """
        Compute gradient with flipped labels.

        Args:
            model: PyTorch model
            data: Input data
            labels: Original labels
            criterion: Loss function
            device: Device to run on

        Returns:
            Gradients as list of numpy arrays
        """
        model.train()

        # Flip labels
        flipped_labels = self.flip_labels(labels.cpu().numpy())
        flipped_labels = torch.tensor(flipped_labels, dtype=torch.long, device=device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, flipped_labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Extract gradients
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.cpu().numpy().copy())

        return gradients

    def get_attack_success_rate(
        self,
        model: nn.Module,
        data: torch.Tensor,
        true_labels: torch.Tensor,
        device: str = "cpu"
    ) -> float:
        """
        Measure attack success: rate of misclassification on flipped labels.

        Args:
            model: Trained model
            data: Test data
            true_labels: True labels
            device: Device to run on

        Returns:
            Attack success rate (fraction of misclassified flipped samples)
        """
        model.eval()

        with torch.no_grad():
            outputs = model(data)
            predictions = torch.argmax(outputs, dim=1)

        # Flip labels to see what model predicts
        flipped_labels = self.flip_labels(true_labels.cpu().numpy())
        flipped_labels = torch.tensor(flipped_labels, device=device)

        # Attack success: model predicts flipped label
        # (i.e., model learned the flipped relationship)
        success = (predictions == flipped_labels).float().mean().item()

        return success
