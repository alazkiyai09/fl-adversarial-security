"""
Deep Leakage from Gradients (DLG) attack using L-BFGS optimizer.
Original implementation of Zhu et al., NeurIPS 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from .base_attack import GradientLeakageAttack, ReconstructionResult


class DLGAttack(GradientLeakageAttack):
    """
    Deep Leakage from Gradients (DLG) attack.

    Uses L-BFGS optimizer to minimize gradient distance.
    Most effective for small batch sizes (batch_size=1).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize DLG attack.

        Args:
            model: Target model
            criterion: Loss function
            device: Device to use
        """
        super().__init__(model, criterion, device)

    def reconstruct(
        self,
        true_gradients: Dict[str, torch.Tensor],
        input_shape: Tuple[int, ...],
        num_classes: int,
        num_iterations: int = 1000,
        init_method: str = 'uniform',
        distance_metric: str = 'mse',
        verbose: bool = True,
        **kwargs
    ) -> ReconstructionResult:
        """
        Reconstruct data using DLG with L-BFGS.

        Args:
            true_gradients: Ground-truth gradients to match
            input_shape: Shape of input data (without batch dimension)
            num_classes: Number of classes
            num_iterations: Maximum L-BFGS iterations
            init_method: Initialization method for dummy data
            distance_metric: Gradient distance metric ('mse', 'cosine')
            verbose: Whether to print progress
            **kwargs: Additional parameters (unused)

        Returns:
            ReconstructionResult
        """
        # Move true gradients to device
        true_gradients = {k: v.to(self.device) for k, v in true_gradients.items()}

        # Initialize dummy data
        batch_size = 1  # DLG typically uses batch_size=1
        dummy_x, dummy_y = self.initialize_dummy_data(
            input_shape, num_classes, batch_size, init_method
        )

        # Track optimization history
        gradient_distances = []

        # Define closure for L-BFGS
        def closure():
            # Zero gradients
            if dummy_x.grad is not None:
                dummy_x.grad.data.zero_()
            if dummy_y.grad is not None:
                dummy_y.grad.data.zero_()

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

            gradient_distances.append(matching_loss.item())

            if verbose and len(gradient_distances) % 100 == 0:
                print(f"  Iteration {len(gradient_distances)}: Loss = {matching_loss.item():.6f}")

            return matching_loss

        # Optimize with L-BFGS
        optimizer = torch.optim.LBFGS(
            [dummy_x, dummy_y],
            max_iter=num_iterations,
            tolerance_change=1e-9,
            history_size=100
        )

        if verbose:
            print(f"Starting L-BFGS optimization (max {num_iterations} iterations)...")

        optimizer.step(closure)

        if verbose:
            print(f"Optimization completed. Final loss: {gradient_distances[-1]:.6f}")

        # Process results
        reconstructed_x = dummy_x.detach().cpu()

        # Get reconstructed label
        if dummy_y.dim() > 1:
            # Soft labels - convert to hard labels
            reconstructed_y = dummy_y.argmax(dim=1).detach().cpu()
        else:
            reconstructed_y = dummy_y.round().long().detach().cpu()

        # Clamp reconstructed data to valid range [0, 1]
        reconstructed_x = torch.clamp(reconstructed_x, 0, 1)

        final_matching_loss = gradient_distances[-1]
        convergence_iterations = len(gradient_distances)

        # Determine success (gradient distance < threshold)
        success = final_matching_loss < 1.0  # Relaxed threshold

        return ReconstructionResult(
            reconstructed_x=reconstructed_x,
            reconstructed_y=reconstructed_y,
            final_matching_loss=final_matching_loss,
            gradient_distances=gradient_distances,
            convergence_iterations=convergence_iterations,
            success=success,
            metadata={
                'optimizer': 'lbfgs',
                'distance_metric': distance_metric,
                'init_method': init_method
            }
        )


def dlg_lbfgs(
    true_gradients: Dict[str, torch.Tensor],
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_classes: int,
    num_iterations: int = 1000,
    init_method: str = 'uniform',
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> ReconstructionResult:
    """
    Convenience function for DLG attack with L-BFGS.

    Args:
        true_gradients: Ground-truth gradients
        model: Target model
        input_shape: Shape of input data (without batch dimension)
        num_classes: Number of classes
        num_iterations: Maximum iterations
        init_method: Initialization method
        device: Device to use
        verbose: Whether to print progress

    Returns:
        ReconstructionResult
    """
    attack = DLGAttack(model, device=device)
    return attack.reconstruct(
        true_gradients=true_gradients,
        input_shape=input_shape,
        num_classes=num_classes,
        num_iterations=num_iterations,
        init_method=init_method,
        verbose=verbose
    )


def dlg_with_multiple_restarts(
    true_gradients: Dict[str, torch.Tensor],
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_classes: int,
    num_restarts: int = 10,
    num_iterations: int = 1000,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> ReconstructionResult:
    """
    DLG attack with multiple random restarts.

    Tries multiple random initializations and returns the best result.

    Args:
        true_gradients: Ground-truth gradients
        model: Target model
        input_shape: Shape of input data
        num_classes: Number of classes
        num_restarts: Number of random restarts
        num_iterations: Iterations per restart
        device: Device to use
        verbose: Whether to print progress

    Returns:
        Best ReconstructionResult across restarts
    """
    attack = DLGAttack(model, device=device)

    best_result = None
    best_loss = float('inf')

    if verbose:
        print(f"\nRunning DLG with {num_restarts} random restarts...")

    for restart in range(num_restarts):
        if verbose:
            print(f"\n--- Restart {restart + 1}/{num_restarts} ---")

        # Set random seed for this restart
        torch.manual_seed(restart)

        result = attack.reconstruct(
            true_gradients=true_gradients,
            input_shape=input_shape,
            num_classes=num_classes,
            num_iterations=num_iterations,
            init_method='uniform',
            verbose=False
        )

        if verbose:
            print(f"Final loss: {result.final_matching_loss:.6f}")

        # Keep best result
        if result.final_matching_loss < best_loss:
            best_loss = result.final_matching_loss
            best_result = result
            best_result.metadata['restart_used'] = restart

    if verbose:
        print(f"\nBest result from restart {best_result.metadata['restart_used']}:")
        print(f"  Final loss: {best_result.final_matching_loss:.6f}")

    return best_result


if __name__ == "__main__":
    # Test DLG attack
    from models.simple_cnn import SimpleCNN
    from data.preparation import prepare_ground_truth_gradients

    print("Testing DLG attack...")

    # Setup
    device = torch.device('cpu')
    model = SimpleCNN(input_channels=1, num_classes=10).to(device)

    # Create ground truth
    true_x = torch.rand(1, 1, 28, 28).to(device)
    true_y = torch.tensor([3]).to(device)

    print(f"\nTrue label: {true_y.item()}")

    # Compute gradients
    true_gradients, loss = prepare_ground_truth_gradients(
        model, true_x, true_y, device=device
    )
    print(f"True loss: {loss:.4f}")

    # Run attack
    result = dlg_lbfgs(
        true_gradients=true_gradients,
        model=model,
        input_shape=(1, 28, 28),
        num_classes=10,
        num_iterations=500,
        device=device,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Reconstructed label: {result.reconstructed_y.item()}")
    print(f"  Label match: {result.reconstructed_y.item() == true_y.item()}")
    print(f"  Final matching loss: {result.final_matching_loss:.6f}")
    print(f"  Data MSE: {F.mse_loss(result.reconstructed_x, true_x):.6f}")
