"""
Differential Privacy noise mechanisms as defenses against gradient leakage.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class DPDefense:
    """
    Add Differential Privacy noise to gradients.
    """

    def __init__(
        self,
        noise_type: str = 'gaussian',
        sigma: float = 1.0,
        sensitivity: float = 1.0,
        clip_norm: Optional[float] = None
    ):
        """
        Initialize DP defense.

        Args:
            noise_type: Type of noise ('gaussian', 'laplace')
            sigma: Noise scale (standard deviation for Gaussian, scale for Laplace)
            sensitivity: Gradient sensitivity (for DP calculation)
            clip_norm: Optional gradient clipping norm
        """
        self.noise_type = noise_type
        self.sigma = sigma
        self.sensitivity = sensitivity
        self.clip_norm = clip_norm

    def add_noise(
        self,
        gradients: Dict[str, torch.Tensor],
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Add noise to gradients.

        Args:
            gradients: Gradient dictionary
            seed: Random seed for reproducibility

        Returns:
            Noisy gradient dictionary
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        noisy_gradients = {}

        for name, grad in gradients.items():
            # Optional gradient clipping
            if self.clip_norm is not None:
                grad = self._clip_gradient(grad, self.clip_norm)

            # Add noise
            if self.noise_type == 'gaussian':
                noise = torch.randn_like(grad) * self.sigma * self.sensitivity
            elif self.noise_type == 'laplace':
                noise = torch.distributions.Laplace(
                    torch.zeros_like(grad),
                    torch.ones_like(grad) * self.sigma * self.sensitivity
                ).sample()
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")

            noisy_gradients[name] = grad + noise

        return noisy_gradients

    def _clip_gradient(self, grad: torch.Tensor, max_norm: float) -> torch.Tensor:
        """Clip gradient by norm."""
        grad_norm = torch.norm(grad)
        if grad_norm > max_norm:
            grad = grad * (max_norm / grad_norm)
        return grad

    def compute_epsilon(
        self,
        num_steps: int = 1,
        delta: float = 1e-5
    ) -> float:
        """
        Compute approximate privacy parameter epsilon.

        Args:
            num_steps: Number of training steps
            delta: Delta parameter for (epsilon, delta)-DP

        Returns:
            Epsilon value
        """
        if self.noise_type == 'gaussian':
            # Gaussian mechanism
            epsilon = num_steps * self.sensitivity / (self.sigma * np.sqrt(2 * np.log(1.25 / delta)))
        elif self.noise_type == 'laplace':
            # Laplace mechanism
            epsilon = num_steps * self.sensitivity / self.sigma
        else:
            epsilon = float('inf')

        return epsilon


class AdaptiveDPDefense(DPDefense):
    """
    Adaptive DP defense that adjusts noise based on gradient magnitude.
    """

    def __init__(
        self,
        noise_type: str = 'gaussian',
        sigma: float = 1.0,
        sensitivity: float = 1.0,
        target_signal_to_noise: float = 10.0
    ):
        """
        Initialize adaptive DP defense.

        Args:
            noise_type: Type of noise
            sigma: Base noise scale
            sensitivity: Gradient sensitivity
            target_signal_to_noise: Target signal-to-noise ratio
        """
        super().__init__(noise_type, sigma, sensitivity)
        self.target_stnr = target_signal_to_noise

    def add_noise(
        self,
        gradients: Dict[str, torch.Tensor],
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Add adaptive noise to gradients.

        Noise scale is proportional to gradient magnitude to maintain
        consistent signal-to-noise ratio.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        noisy_gradients = {}

        for name, grad in gradients.items():
            # Compute adaptive noise scale
            grad_norm = torch.norm(grad).item()
            adaptive_sigma = max(self.sigma, grad_norm / self.target_stnr)

            # Add noise
            if self.noise_type == 'gaussian':
                noise = torch.randn_like(grad) * adaptive_sigma * self.sensitivity
            elif self.noise_type == 'laplace':
                noise = torch.distributions.Laplace(
                    torch.zeros_like(grad),
                    torch.ones_like(grad) * adaptive_sigma * self.sensitivity
                ).sample()
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")

            noisy_gradients[name] = grad + noise

        return noisy_gradients


def test_dp_defense_effectiveness(
    model: torch.nn.Module,
    attack_fn,
    defense: DPDefense,
    test_samples: list,
    sigma_values: list,
    device: torch.device = torch.device('cpu')
) -> Tuple[list, list]:
    """
    Test DP defense effectiveness across different noise levels.

    Args:
        model: Target model
        attack_fn: Attack function to use
        defense: DP defense instance
        test_samples: List of (x, y) tuples to test
        sigma_values: List of sigma values to test
        device: Device to use

    Returns:
        (label_accuracies, mse_values) tuples
    """
    from ..data.preparation import compute_gradients
    from ..metrics.reconstruction_quality import compute_reconstruction_metrics

    label_accuracies = []
    mse_values = []

    for sigma in sigma_values:
        defense.sigma = sigma

        correct_labels = 0
        mse_sum = 0

        for x, y in test_samples:
            x = x.to(device)
            y = y.to(device)

            # Compute gradients
            true_gradients = compute_gradients(model, x, y)

            # Add noise
            noisy_gradients = defense.add_noise(true_gradients)

            # Run attack
            result = attack_fn(
                noisy_gradients,
                model,
                input_shape=x.shape[1:],
                num_classes=10
            )

            # Evaluate
            if result.reconstructed_y.item() == y.item():
                correct_labels += 1

            mse = torch.nn.functional.mse_loss(result.reconstructed_x, x).item()
            mse_sum += mse

        label_accuracy = correct_labels / len(test_samples)
        avg_mse = mse_sum / len(test_samples)

        label_accuracies.append(label_accuracy)
        mse_values.append(avg_mse)

        print(f"  Sigma={sigma:.3f}: Label Acc={label_accuracy:.2%}, MSE={avg_mse:.6f}")

    return label_accuracies, mse_values


if __name__ == "__main__":
    # Test DP defense
    print("Testing DP defense...")

    # Create dummy gradients
    gradients = {
        'weight1': torch.randn(10, 10),
        'bias1': torch.randn(10),
        'weight2': torch.randn(10, 5)
    }

    # Test Gaussian noise
    print("\nGaussian noise (sigma=0.5):")
    defense = DPDefense(noise_type='gaussian', sigma=0.5)
    noisy_grads = defense.add_noise(gradients)

    for name in gradients.keys():
        orig_norm = torch.norm(gradients[name]).item()
        noisy_norm = torch.norm(noisy_grads[name]).item()
        diff = torch.norm(gradients[name] - noisy_grads[name]).item()
        print(f"  {name}: orig_norm={orig_norm:.4f}, noisy_norm={noisy_norm:.4f}, diff={diff:.4f}")

    # Test Laplace noise
    print("\nLaplace noise (sigma=0.5):")
    defense = DPDefense(noise_type='laplace', sigma=0.5)
    noisy_grads = defense.add_noise(gradients)

    for name in gradients.keys():
        orig_norm = torch.norm(gradients[name]).item()
        noisy_norm = torch.norm(noisy_grads[name]).item()
        diff = torch.norm(gradients[name] - noisy_grads[name]).item()
        print(f"  {name}: orig_norm={orig_norm:.4f}, noisy_norm={noisy_norm:.4f}, diff={diff:.4f}")

    # Test epsilon computation
    print("\nPrivacy parameters:")
    for sigma in [0.1, 0.5, 1.0, 2.0]:
        defense = DPDefense(sigma=sigma)
        epsilon = defense.compute_epsilon(num_steps=100)
        print(f"  Sigma={sigma:.1f}: Epsilon≈{epsilon:.4f}")
