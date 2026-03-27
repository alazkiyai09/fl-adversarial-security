"""
DLG attack using Adam optimizer.
More stable than L-BFGS for some cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from .base_attack import GradientLeakageAttack, ReconstructionResult


class DLGAdamAttack(GradientLeakageAttack):
    """
    DLG attack with Adam optimizer.

    Adam can be more stable than L-BFGS, especially for:
    - Larger batch sizes
    - More complex models
    - Noisy gradient scenarios
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        device: torch.device = torch.device('cpu')
    ):
        """Initialize DLG-Adam attack."""
        super().__init__(model, criterion, device)

    def reconstruct(
        self,
        true_gradients: Dict[str, torch.Tensor],
        input_shape: Tuple[int, ...],
        num_classes: int,
        num_iterations: int = 2000,
        lr: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        init_method: str = 'uniform',
        distance_metric: str = 'mse',
        verbose: bool = True,
        log_interval: int = 100,
        **kwargs
    ) -> ReconstructionResult:
        """
        Reconstruct data using DLG with Adam optimizer.

        Args:
            true_gradients: Ground-truth gradients to match
            input_shape: Shape of input data
            num_classes: Number of classes
            num_iterations: Number of optimization steps
            lr: Learning rate
            betas: Adam betas
            init_method: Initialization method
            distance_metric: Gradient distance metric
            verbose: Whether to print progress
            log_interval: Log every N iterations
            **kwargs: Additional parameters

        Returns:
            ReconstructionResult
        """
        # Move true gradients to device
        true_gradients = {k: v.to(self.device) for k, v in true_gradients.items()}

        # Initialize dummy data
        batch_size = 1
        dummy_x, dummy_y = self.initialize_dummy_data(
            input_shape, num_classes, batch_size, init_method
        )

        # Create optimizer
        optimizer = torch.optim.Adam(
            [dummy_x, dummy_y],
            lr=lr,
            betas=betas
        )

        # Learning rate scheduler (optional)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iterations
        )

        # Track optimization history
        gradient_distances = []

        if verbose:
            print(f"Starting Adam optimization ({num_iterations} iterations, lr={lr})...")

        # Optimization loop
        for iteration in range(num_iterations):
            # Zero gradients
            optimizer.zero_grad()

            # Compute dummy gradients
            dummy_gradients = self.compute_dummy_gradients(
                dummy_x, dummy_y, create_graph=True
            )

            # Compute matching loss
            matching_loss = self.compute_gradient_distance(
                dummy_gradients, true_gradients, metric=distance_metric
            )

            # Backward
            matching_loss.backward()

            # Update
            optimizer.step()
            scheduler.step()

            # Track
            gradient_distances.append(matching_loss.item())

            # Logging
            if verbose and iteration % log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Iteration {iteration}: Loss = {matching_loss.item():.6f}, LR = {current_lr:.6f}")

            # Optional: clamp data to valid range
            with torch.no_grad():
                dummy_x.data.clamp_(0, 1)

        if verbose:
            print(f"Optimization completed. Final loss: {gradient_distances[-1]:.6f}")

        # Process results
        reconstructed_x = dummy_x.detach().cpu()
        reconstructed_x = torch.clamp(reconstructed_x, 0, 1)

        if dummy_y.dim() > 1:
            reconstructed_y = dummy_y.argmax(dim=1).detach().cpu()
        else:
            reconstructed_y = dummy_y.round().long().detach().cpu()

        final_matching_loss = gradient_distances[-1]
        convergence_iterations = len(gradient_distances)

        success = final_matching_loss < 1.0

        return ReconstructionResult(
            reconstructed_x=reconstructed_x,
            reconstructed_y=reconstructed_y,
            final_matching_loss=final_matching_loss,
            gradient_distances=gradient_distances,
            convergence_iterations=convergence_iterations,
            success=success,
            metadata={
                'optimizer': 'adam',
                'lr': lr,
                'betas': betas,
                'distance_metric': distance_metric,
                'init_method': init_method
            }
        )


def dlg_adam(
    true_gradients: Dict[str, torch.Tensor],
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_classes: int,
    num_iterations: int = 2000,
    lr: float = 0.1,
    init_method: str = 'uniform',
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> ReconstructionResult:
    """
    Convenience function for DLG with Adam.

    Args:
        true_gradients: Ground-truth gradients
        model: Target model
        input_shape: Shape of input data
        num_classes: Number of classes
        num_iterations: Number of iterations
        lr: Learning rate
        init_method: Initialization method
        device: Device to use
        verbose: Whether to print progress

    Returns:
        ReconstructionResult
    """
    attack = DLGAdamAttack(model, device=device)
    return attack.reconstruct(
        true_gradients=true_gradients,
        input_shape=input_shape,
        num_classes=num_classes,
        num_iterations=num_iterations,
        lr=lr,
        init_method=init_method,
        verbose=verbose
    )


def dlg_adam_with_warmup(
    true_gradients: Dict[str, torch.Tensor],
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_classes: int,
    num_iterations: int = 2000,
    warmup_iterations: int = 100,
    init_lr: float = 0.01,
    max_lr: float = 0.5,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> ReconstructionResult:
    """
    DLG with Adam and learning rate warmup.

    Gradually increases learning rate, then uses cosine decay.

    Args:
        true_gradients: Ground-truth gradients
        model: Target model
        input_shape: Shape of input data
        num_classes: Number of classes
        num_iterations: Total iterations
        warmup_iterations: Warmup period
        init_lr: Initial learning rate
        max_lr: Maximum learning rate
        device: Device to use
        verbose: Whether to print progress

    Returns:
        ReconstructionResult
    """
    attack = DLGAdamAttack(model, device=device)

    # Move gradients to device
    true_gradients = {k: v.to(device) for k, v in true_gradients.items()}

    # Initialize dummy data
    batch_size = 1
    dummy_x, dummy_y = attack.initialize_dummy_data(
        input_shape, num_classes, batch_size, 'uniform'
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        [dummy_x, dummy_y],
        lr=init_lr
    )

    # Custom scheduler with warmup
    def get_lr(iteration):
        if iteration < warmup_iterations:
            # Linear warmup
            return init_lr + (max_lr - init_lr) * iteration / warmup_iterations
        else:
            # Cosine decay
            progress = (iteration - warmup_iterations) / (num_iterations - warmup_iterations)
            return init_lr + (max_lr - init_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()

    # Track
    gradient_distances = []

    if verbose:
        print(f"Starting Adam with warmup ({num_iterations} iterations)...")

    for iteration in range(num_iterations):
        # Set learning rate
        current_lr = get_lr(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Zero gradients
        optimizer.zero_grad()

        # Compute loss
        dummy_gradients = attack.compute_dummy_gradients(dummy_x, dummy_y, create_graph=True)
        matching_loss = attack.compute_gradient_distance(
            dummy_gradients, true_gradients, metric='mse'
        )

        # Backward
        matching_loss.backward()
        optimizer.step()

        # Clamp data
        with torch.no_grad():
            dummy_x.data.clamp_(0, 1)

        gradient_distances.append(matching_loss.item())

        if verbose and iteration % 100 == 0:
            print(f"  Iteration {iteration}: Loss = {matching_loss.item():.6f}, LR = {current_lr:.4f}")

    if verbose:
        print(f"Optimization completed. Final loss: {gradient_distances[-1]:.6f}")

    # Process results
    reconstructed_x = torch.clamp(dummy_x.detach().cpu(), 0, 1)
    reconstructed_y = dummy_y.argmax(dim=1) if dummy_y.dim() > 1 else dummy_y.round().long()
    reconstructed_y = reconstructed_y.detach().cpu()

    return ReconstructionResult(
        reconstructed_x=reconstructed_x,
        reconstructed_y=reconstructed_y,
        final_matching_loss=gradient_distances[-1],
        gradient_distances=gradient_distances,
        convergence_iterations=len(gradient_distances),
        success=gradient_distances[-1] < 1.0,
        metadata={
            'optimizer': 'adam_warmup',
            'warmup_iterations': warmup_iterations,
            'max_lr': max_lr
        }
    )


if __name__ == "__main__":
    # Test DLG-Adam
    from models.simple_cnn import SimpleCNN
    from data.preparation import prepare_ground_truth_gradients

    print("Testing DLG with Adam...")

    device = torch.device('cpu')
    model = SimpleCNN(input_channels=1, num_classes=10).to(device)

    true_x = torch.rand(1, 1, 28, 28).to(device)
    true_y = torch.tensor([7]).to(device)

    print(f"\nTrue label: {true_y.item()}")

    true_gradients, loss = prepare_ground_truth_gradients(model, true_x, true_y, device=device)
    print(f"True loss: {loss:.4f}")

    # Run attack
    result = dlg_adam(
        true_gradients=true_gradients,
        model=model,
        input_shape=(1, 28, 28),
        num_classes=10,
        num_iterations=1000,
        lr=0.1,
        device=device,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Reconstructed label: {result.reconstructed_y.item()}")
    print(f"  Label match: {result.reconstructed_y.item() == true_y.item()}")
    print(f"  Final matching loss: {result.final_matching_loss:.6f}")
    print(f"  Data MSE: {F.mse_loss(result.reconstructed_x, true_x):.6f}")
