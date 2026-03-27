"""
Improved DLG using cosine similarity loss.
Often converges faster and more stably than MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from .base_attack import GradientLeakageAttack, ReconstructionResult


class DLGCosineAttack(GradientLeakageAttack):
    """
    DLG attack with cosine similarity loss.

    Uses cosine similarity instead of MSE for gradient matching.
    Can be more stable and converge faster.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        device: torch.device = torch.device('cpu')
    ):
        """Initialize DLG-Cosine attack."""
        super().__init__(model, criterion, device)

    def reconstruct(
        self,
        true_gradients: Dict[str, torch.Tensor],
        input_shape: Tuple[int, ...],
        num_classes: int,
        num_iterations: int = 2000,
        lr: float = 0.01,
        momentum: float = 0.9,
        init_method: str = 'uniform',
        use_magnitude_matching: bool = True,
        verbose: bool = True,
        log_interval: int = 100,
        **kwargs
    ) -> ReconstructionResult:
        """
        Reconstruct data using DLG with cosine similarity loss.

        Args:
            true_gradients: Ground-truth gradients to match
            input_shape: Shape of input data
            num_classes: Number of classes
            num_iterations: Number of optimization steps
            lr: Learning rate
            momentum: Momentum for SGD optimizer
            init_method: Initialization method
            use_magnitude_matching: Whether to also match gradient magnitude
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

        # Create optimizer (SGD with momentum works well with cosine loss)
        optimizer = torch.optim.SGD(
            [dummy_x, dummy_y],
            lr=lr,
            momentum=momentum
        )

        # Flatten true gradients for cosine computation
        true_grad_vec = torch.cat([g.flatten() for g in true_gradients.values()])
        true_grad_norm = torch.norm(true_grad_vec)

        # Track optimization history
        gradient_distances = []

        if verbose:
            print(f"Starting cosine similarity optimization ({num_iterations} iterations)...")
            if use_magnitude_matching:
                print("  Using combined cosine + magnitude matching")

        # Optimization loop
        for iteration in range(num_iterations):
            # Zero gradients
            optimizer.zero_grad()

            # Compute dummy gradients
            dummy_gradients = self.compute_dummy_gradients(
                dummy_x, dummy_y, create_graph=True
            )

            # Flatten dummy gradients
            dummy_grad_vec = torch.cat([g.flatten() for g in dummy_gradients.values()])
            dummy_grad_norm = torch.norm(dummy_grad_vec)

            # Compute cosine similarity loss
            cosine_sim = F.cosine_similarity(dummy_grad_vec.unsqueeze(0), true_grad_vec.unsqueeze(0))
            cosine_loss = 1 - cosine_sim  # Convert to loss (lower is better)

            # Optionally also match magnitude
            if use_magnitude_matching:
                magnitude_loss = (dummy_grad_norm - true_grad_norm) ** 2 / (true_grad_norm ** 2)
                matching_loss = cosine_loss + 0.1 * magnitude_loss
            else:
                matching_loss = cosine_loss

            # Backward
            matching_loss.backward()

            # Update
            optimizer.step()

            # Clamp data to valid range
            with torch.no_grad():
                dummy_x.data.clamp_(0, 1)

            # Track
            gradient_distances.append(matching_loss.item())

            # Logging
            if verbose and iteration % log_interval == 0:
                print(f"  Iteration {iteration}: Loss = {matching_loss.item():.6f}, "
                      f"Cosine = {cosine_sim.item():.4f}, "
                      f"Mag ratio = {(dummy_grad_norm / true_grad_norm).item():.4f}")

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

        success = final_matching_loss < 0.1  # Cosine loss threshold

        return ReconstructionResult(
            reconstructed_x=reconstructed_x,
            reconstructed_y=reconstructed_y,
            final_matching_loss=final_matching_loss,
            gradient_distances=gradient_distances,
            convergence_iterations=convergence_iterations,
            success=success,
            metadata={
                'optimizer': 'sgd_momentum',
                'lr': lr,
                'momentum': momentum,
                'use_magnitude_matching': use_magnitude_matching,
                'init_method': init_method
            }
        )


class ImprovedDLG(GradientLeakageAttack):
    """
    Improved DLG with adaptive loss and optimization strategy.

    Combines multiple techniques:
    - Cosine similarity loss
    - Adaptive learning rate
    - Label optimization strategies
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        device: torch.device = torch.device('cpu')
    ):
        """Initialize Improved DLG."""
        super().__init__(model, criterion, device)

    def reconstruct(
        self,
        true_gradients: Dict[str, torch.Tensor],
        input_shape: Tuple[int, ...],
        num_classes: int,
        num_iterations: int = 2000,
        lr: float = 0.1,
        init_method: str = 'uniform',
        label_strategy: str = 'soft',
        verbose: bool = True,
        **kwargs
    ) -> ReconstructionResult:
        """
        Reconstruct data using improved DLG.

        Args:
            true_gradients: Ground-truth gradients
            input_shape: Shape of input data
            num_classes: Number of classes
            num_iterations: Number of optimization steps
            lr: Learning rate
            init_method: Initialization method
            label_strategy: 'soft' (continuous) or 'hard' (discrete)
            verbose: Whether to print progress
            **kwargs: Additional parameters

        Returns:
            ReconstructionResult
        """
        true_gradients = {k: v.to(self.device) for k, v in true_gradients.items()}

        batch_size = 1
        dummy_x, dummy_y = self.initialize_dummy_data(
            input_shape, num_classes, batch_size, init_method
        )

        # For soft label strategy, use one-hot encoding
        if label_strategy == 'soft':
            # Initialize as uniform distribution
            dummy_y = torch.ones(batch_size, num_classes, device=self.device) / num_classes
            dummy_y.requires_grad = True

        # Adam optimizer
        optimizer = torch.optim.Adam(
            [dummy_x, dummy_y],
            lr=lr
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iterations
        )

        # Flatten true gradients
        true_grad_vec = torch.cat([g.flatten() for g in true_gradients.values()])

        gradient_distances = []

        if verbose:
            print(f"Starting Improved DLG ({label_strategy} labels, {num_iterations} iterations)...")

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Compute dummy gradients
            dummy_output = self.model(dummy_x)

            # Handle different label strategies
            if label_strategy == 'soft':
                # Soft labels: use cross-entropy with soft targets
                log_probs = F.log_softmax(dummy_output, dim=1)
                loss = -(log_probs * dummy_y).sum(dim=1).mean()
            else:
                # Hard labels
                loss = F.cross_entropy(dummy_output, dummy_y.argmax(dim=1))

            # Compute gradients
            dummy_gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param.grad.data.zero_()

            loss.backward(create_graph=True)

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    dummy_gradients[name] = param.grad.clone()

            # Clear model gradients
            for param in self.model.parameters():
                param.grad = None

            # Compute matching loss (cosine)
            dummy_grad_vec = torch.cat([g.flatten() for g in dummy_gradients.values()])
            cosine_sim = F.cosine_similarity(dummy_grad_vec.unsqueeze(0), true_grad_vec.unsqueeze(0))
            matching_loss = 1 - cosine_sim

            matching_loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                dummy_x.data.clamp_(0, 1)
                if label_strategy == 'soft':
                    # Normalize soft labels to sum to 1
                    dummy_y.data = F.softmax(dummy_y.data, dim=1)

            gradient_distances.append(matching_loss.item())

            if verbose and iteration % 100 == 0:
                print(f"  Iteration {iteration}: Loss = {matching_loss.item():.6f}, "
                      f"Cosine = {cosine_sim.item():.4f}")

        if verbose:
            print(f"Optimization completed. Final loss: {gradient_distances[-1]:.6f}")

        # Process results
        reconstructed_x = torch.clamp(dummy_x.detach().cpu(), 0, 1)

        if label_strategy == 'soft':
            reconstructed_y = dummy_y.argmax(dim=1).detach().cpu()
        else:
            reconstructed_y = dummy_y.argmax(dim=1) if dummy_y.dim() > 1 else dummy_y.round().long()
            reconstructed_y = reconstructed_y.detach().cpu()

        return ReconstructionResult(
            reconstructed_x=reconstructed_x,
            reconstructed_y=reconstructed_y,
            final_matching_loss=gradient_distances[-1],
            gradient_distances=gradient_distances,
            convergence_iterations=len(gradient_distances),
            success=gradient_distances[-1] < 0.1,
            metadata={
                'label_strategy': label_strategy,
                'lr': lr
            }
        )


def dlg_cosine(
    true_gradients: Dict[str, torch.Tensor],
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_classes: int,
    num_iterations: int = 2000,
    lr: float = 0.01,
    use_magnitude_matching: bool = True,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> ReconstructionResult:
    """
    Convenience function for DLG with cosine similarity.

    Args:
        true_gradients: Ground-truth gradients
        model: Target model
        input_shape: Shape of input data
        num_classes: Number of classes
        num_iterations: Number of iterations
        lr: Learning rate
        use_magnitude_matching: Whether to match gradient magnitude
        device: Device to use
        verbose: Whether to print progress

    Returns:
        ReconstructionResult
    """
    attack = DLGCosineAttack(model, device=device)
    return attack.reconstruct(
        true_gradients=true_gradients,
        input_shape=input_shape,
        num_classes=num_classes,
        num_iterations=num_iterations,
        lr=lr,
        use_magnitude_matching=use_magnitude_matching,
        verbose=verbose
    )


if __name__ == "__main__":
    # Test DLG-Cosine
    from models.simple_cnn import SimpleCNN
    from data.preparation import prepare_ground_truth_gradients

    print("Testing DLG with Cosine Similarity...")

    device = torch.device('cpu')
    model = SimpleCNN(input_channels=1, num_classes=10).to(device)

    true_x = torch.rand(1, 1, 28, 28).to(device)
    true_y = torch.tensor([5]).to(device)

    print(f"\nTrue label: {true_y.item()}")

    true_gradients, loss = prepare_ground_truth_gradients(model, true_x, true_y, device=device)
    print(f"True loss: {loss:.4f}")

    # Run attack
    result = dlg_cosine(
        true_gradients=true_gradients,
        model=model,
        input_shape=(1, 28, 28),
        num_classes=10,
        num_iterations=1000,
        lr=0.01,
        device=device,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Reconstructed label: {result.reconstructed_y.item()}")
    print(f"  Label match: {result.reconstructed_y.item() == true_y.item()}")
    print(f"  Final matching loss: {result.final_matching_loss:.6f}")
    print(f"  Data MSE: {F.mse_loss(result.reconstructed_x, true_x):.6f}")
