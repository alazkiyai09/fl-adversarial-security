"""Differential Privacy mechanisms for federated learning.

Implements DP-SGD and related privacy mechanisms:
- Gradient clipping
- Gaussian noise addition
- Privacy accounting (ε, δ tracking)
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from loguru import logger


@dataclass
class PrivacySpent:
    """Track privacy budget consumption."""
    epsilon: float
    delta: float
    round: int


def clip_and_add_noise(
    gradients: List[torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    sigma: Optional[float] = None,
) -> List[torch.Tensor]:
    """
    Apply gradient clipping and Gaussian noise (DP-SGD).

    Args:
        gradients: List of gradient tensors
        clip_norm: Maximum L2 norm for clipping
        noise_multiplier: Multiplier for noise standard deviation
        sigma: Standard deviation (computed from clip_norm * noise_multiplier if None)

    Returns:
        List of clipped and noised gradients
    """
    if sigma is None:
        sigma = clip_norm * noise_multiplier

    clipped_gradients = []

    # Flatten and compute total norm
    flat_grads = [g.flatten() for g in gradients]
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in flat_grads))

    # Clip gradients
    if total_norm > clip_norm:
        clip_factor = clip_norm / total_norm
        clipped_grads = [g * clip_factor for g in gradients]
    else:
        clipped_grads = gradients

    # Add noise
    noised_grads = []
    for grad in clipped_grads:
        noise = torch.randn_like(grad) * sigma
        noised_grads.append(grad + noise)

    return noised_grads


def compute_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    num_steps: int,
    sample_rate: float,
    epochs: int,
) -> float:
    """
    Compute required noise multiplier for target privacy budget.

    Uses the moments accountant approach.

    Args:
        target_epsilon: Target ε value
        target_delta: Target δ value
        num_steps: Number of training steps
        sample_rate: Data sampling probability
        epochs: Number of training epochs

    Returns:
        Noise multiplier (σ)
    """
    try:
        from opacus.accountants.analysis import rdp as privacy_analysis
    except ImportError:
        logger.warning("Opacus not available, using default noise multiplier")
        return 1.0

    # Binary search for noise multiplier
    low, high = 0.1, 100.0

    for _ in range(30):  # 30 iterations should be sufficient
        mid = (low + high) / 2

        # Compute epsilon with current noise multiplier
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = privacy_analysis.compute_rdp(
            sample_rate=sample_rate,
            noise_multiplier=mid,
            steps=num_steps * epochs,
            orders=orders,
        )
        epsilon = privacy_analysis.get_privacy_spent(orders, rdp, target_delta)[0]

        if epsilon < target_epsilon:
            high = mid
        else:
            low = mid

    return (low + high) / 2


def compute_sampling_probability(
    batch_size: int,
    dataset_size: int,
) -> float:
    """
    Compute sampling probability for privacy accounting.

    Args:
        batch_size: Training batch size
        dataset_size: Size of training dataset

    Returns:
        Sampling probability q
    """
    return min(1.0, batch_size / dataset_size)


class DPSGDOptimizer(Optimizer):
    """
    DP-SGD optimizer with gradient clipping and noise injection.

    Wraps any PyTorch optimizer to add differential privacy.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        noise_multiplier: float,
        max_grad_norm: float,
        batch_size: int,
        dataset_size: int,
        loss_reduction: str = "mean",
    ):
        """
        Initialize DP-SGD optimizer.

        Args:
            optimizer: Base optimizer (e.g., Adam, SGD)
            noise_multiplier: Noise multiplier σ
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Training batch size
            dataset_size: Size of training dataset
            loss_reduction: How loss is reduced ('mean' or 'sum')
        """
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.loss_reduction = loss_reduction

        # Compute standard deviation
        self.sigma = max_grad_norm * noise_multiplier

        # Sampling probability
        self.sample_rate = batch_size / dataset_size

        # Track privacy spent
        self.steps = 0
        self.epsilon = 0.0
        self.delta = 1e-5  # Default delta

        logger.info(
            f"DP-SGD initialized: noise_multiplier={noise_multiplier}, "
            f"max_grad_norm={max_grad_norm}, sample_rate={self.sample_rate:.4f}"
        )

    @property
    def param_groups(self):
        """Expose optimizer param groups."""
        return self.optimizer.param_groups

    @property
    def state_dict(self):
        """Expose optimizer state_dict."""
        return self.optimizer.state_dict

    @property
    def state(self):
        """Expose optimizer state."""
        return self.optimizer.state

    def step(self, closure=None):
        """
        Perform a single optimization step with DP.

        Args:
            closure: Optional closure for re-evaluating loss

        Returns:
            Loss value if closure is provided
        """
        # Clip and add noise to gradients
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Clip gradient
                grad = p.grad.data
                grad_norm = grad.norm()

                if grad_norm > self.max_grad_norm:
                    grad = grad * (self.max_grad_norm / grad_norm)

                # Add noise
                noise = torch.randn_like(grad) * self.sigma
                p.grad.data = grad + noise

        # Step the base optimizer
        loss = self.optimizer.step(closure)

        # Update privacy tracking
        self.steps += 1
        self._update_privacy_spent()

        return loss

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    def _update_privacy_spent(self) -> None:
        """Update privacy spent after each step."""
        try:
            from opacus.accountants.analysis import rdp as privacy_analysis
        except ImportError:
            return

        if self.steps == 0:
            return

        # Compute RDP
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = privacy_analysis.compute_rdp(
            sample_rate=self.sample_rate,
            noise_multiplier=self.noise_multiplier,
            steps=self.steps,
            orders=orders,
        )

        # Convert to (ε, δ)
        epsilon = privacy_analysis.get_privacy_spent(orders, rdp, self.delta)[0]
        self.epsilon = epsilon

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy budget spent.

        Returns:
            Tuple of (epsilon, delta)
        """
        return self.epsilon, self.delta

    def add_param_group(self, param_group):
        """Add a param group to the optimizer."""
        self.optimizer.add_param_group(param_group)


class DPSGDFactory:
    """
    Factory for creating DP-SGD optimizers.

    Simplifies creation of differentially private optimizers.
    """

    def __init__(
        self,
        noise_multiplier: float,
        max_grad_norm: float,
        delta: float,
        epsilon: Optional[float] = None,
    ):
        """
        Initialize DP-SGD factory.

        Args:
            noise_multiplier: Noise multiplier σ
            max_grad_norm: Maximum gradient norm for clipping
            delta: Target δ value
            epsilon: Target ε value (if computing noise_multiplier)
        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.target_epsilon = epsilon

    def create_dp_optimizer(
        self,
        parameters: List[nn.Parameter],
        lr: float,
        optimizer_type: str = "adam",
        **optimizer_kwargs
    ) -> DPSGDOptimizer:
        """
        Create a DP-SGD optimizer.

        Args:
            parameters: Model parameters
            lr: Learning rate
            optimizer_type: Base optimizer type ('adam', 'sgd', 'adamw')
            **optimizer_kwargs: Additional arguments for base optimizer

        Returns:
            DPSGDOptimizer instance
        """
        # Create base optimizer
        if optimizer_type.lower() == "adam":
            base_optimizer = torch.optim.Adam(
                parameters,
                lr=lr,
                **optimizer_kwargs,
            )
        elif optimizer_type.lower() == "adamw":
            base_optimizer = torch.optim.AdamW(
                parameters,
                lr=lr,
                **optimizer_kwargs,
            )
        elif optimizer_type.lower() == "sgd":
            base_optimizer = torch.optim.SGD(
                parameters,
                lr=lr,
                **optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Estimate batch size and dataset size from optimizer kwargs
        batch_size = optimizer_kwargs.get("batch_size", 32)
        dataset_size = optimizer_kwargs.get("dataset_size", 10000)

        # Wrap with DP-SGD
        dp_optimizer = DPSGDOptimizer(
            optimizer=base_optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            batch_size=batch_size,
            dataset_size=dataset_size,
        )

        logger.info(f"Created DP-{optimizer_type.upper()} optimizer")

        return dp_optimizer

    def create_dp_sgd(
        self,
        parameters: List[nn.Parameter],
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        dataset_size: int = 10000,
    ) -> DPSGDOptimizer:
        """
        Create DP-SGD optimizer.

        Args:
            parameters: Model parameters
            lr: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay (L2 penalty)
            batch_size: Batch size
            dataset_size: Dataset size

        Returns:
            DPSGDOptimizer instance
        """
        base_optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        return DPSGDOptimizer(
            optimizer=base_optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            batch_size=batch_size,
            dataset_size=dataset_size,
        )

    def create_dp_adam(
        self,
        parameters: List[nn.Parameter],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        batch_size: int = 32,
        dataset_size: int = 10000,
    ) -> DPSGDOptimizer:
        """
        Create DP-Adam optimizer.

        Args:
            parameters: Model parameters
            lr: Learning rate
            betas: Adam coefficients
            weight_decay: Weight decay
            batch_size: Batch size
            dataset_size: Dataset size

        Returns:
            DPSGDOptimizer instance
        """
        base_optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        return DPSGDOptimizer(
            optimizer=base_optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            batch_size=batch_size,
            dataset_size=dataset_size,
        )


class PrivacyAccountant:
    """
    Tracks privacy budget consumption over training.

    Computes and tracks (ε, δ) privacy guarantees.
    """

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
        noise_multiplier: float,
        max_grad_norm: float,
        batch_size: int,
        dataset_size: int,
        epochs: int,
    ):
        """
        Initialize privacy accountant.

        Args:
            target_epsilon: Target ε
            target_delta: Target δ
            noise_multiplier: Noise multiplier σ
            max_grad_norm: Gradient clipping threshold
            batch_size: Training batch size
            dataset_size: Dataset size
            epochs: Number of training epochs
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.epochs = epochs

        # Sampling probability
        self.sample_rate = batch_size / dataset_size

        # Tracking
        self.current_epsilon = 0.0
        self.current_delta = 0.0
        self.steps = 0
        self.history: List[PrivacySpent] = []

        logger.info(
            f"PrivacyAccountant initialized: target ε={target_epsilon}, "
            f"δ={target_delta}, σ={noise_multiplier}"
        )

    def step(self, num_steps: int = 1) -> Tuple[float, float]:
        """
        Update privacy spent after training steps.

        Args:
            num_steps: Number of steps taken

        Returns:
            Tuple of (epsilon, delta) spent so far
        """
        self.steps += num_steps

        # Compute privacy spent
        try:
            from opacus.accountants.analysis import rdp as privacy_analysis

            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

            # Compute RDP
            rdp = privacy_analysis.compute_rdp(
                sample_rate=self.sample_rate,
                noise_multiplier=self.noise_multiplier,
                steps=self.steps,
                orders=orders,
            )

            # Convert to (ε, δ)
            epsilon = privacy_analysis.get_privacy_spent(
                orders, rdp, self.target_delta
            )[0]

            self.current_epsilon = epsilon
            self.current_delta = self.target_delta

        except ImportError:
            logger.warning("Opacus not available, privacy accounting disabled")
            # Fallback: simple approximation
            # ε ≈ steps * sample_rate / noise_multiplier
            self.current_epsilon = self.steps * self.sample_rate / self.noise_multiplier
            self.current_delta = self.target_delta

        # Record history
        self.history.append(PrivacySpent(
            epsilon=self.current_epsilon,
            delta=self.current_delta,
            round=self.steps,
        ))

        return self.current_epsilon, self.current_delta

    def get_budget_spent(self) -> Tuple[float, float]:
        """Get current privacy budget spent."""
        return self.current_epsilon, self.current_delta

    def get_budget_remaining(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        remaining_epsilon = max(0.0, self.target_epsilon - self.current_epsilon)
        remaining_delta = max(0.0, self.target_delta - self.current_delta)
        return remaining_epsilon, remaining_delta

    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.current_epsilon >= self.target_epsilon

    def get_history(self) -> List[PrivacySpent]:
        """Get privacy spending history."""
        return self.history.copy()

    def estimate_steps_for_target(self) -> Optional[int]:
        """
        Estimate number of steps to reach target ε.

        Returns:
            Estimated steps, or None if cannot compute
        """
        try:
            from opacus.accountants.analysis import rdp as privacy_analysis

            # Binary search for steps
            low, high = 1, self.epochs * (self.dataset_size // self.batch_size)

            for _ in range(30):
                mid = (low + high) // 2

                orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                rdp = privacy_analysis.compute_rdp(
                    sample_rate=self.sample_rate,
                    noise_multiplier=self.noise_multiplier,
                    steps=mid,
                    orders=orders,
                )
                epsilon = privacy_analysis.get_privacy_spent(
                    orders, rdp, self.target_delta
                )[0]

                if epsilon < self.target_epsilon:
                    low = mid + 1
                else:
                    high = mid - 1

            return high

        except ImportError:
            # Fallback: simple approximation
            # steps ≈ ε * σ / sample_rate
            return int(self.target_epsilon * self.noise_multiplier / self.sample_rate)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of privacy accounting."""
        remaining_epsilon, remaining_delta = self.get_budget_remaining()

        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "current_epsilon": self.current_epsilon,
            "current_delta": self.current_delta,
            "remaining_epsilon": remaining_epsilon,
            "remaining_delta": remaining_delta,
            "budget_exhausted": self.is_budget_exhausted(),
            "steps": self.steps,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "sample_rate": self.sample_rate,
        }


def compute_dp_params(
    target_epsilon: float = 1.0,
    target_delta: float = 1e-5,
    batch_size: int = 32,
    dataset_size: int = 10000,
    epochs: int = 10,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Compute DP-SGD parameters for target privacy budget.

    Args:
        target_epsilon: Target ε value
        target_delta: Target δ value
        batch_size: Training batch size
        dataset_size: Dataset size
        epochs: Number of training epochs
        max_grad_norm: Gradient clipping threshold

    Returns:
        Dictionary with noise_multiplier and estimated_steps
    """
    num_steps = epochs * (dataset_size // batch_size)
    sample_rate = batch_size / dataset_size

    noise_multiplier = compute_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        num_steps=num_steps,
        sample_rate=sample_rate,
        epochs=epochs,
    )

    return {
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "sample_rate": sample_rate,
        "num_steps": num_steps,
    }
