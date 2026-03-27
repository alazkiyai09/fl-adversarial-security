"""
Gradient compression defenses against gradient leakage.
Includes sparsification, quantization, and other compression techniques.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class GradientCompression:
    """
    Base class for gradient compression defenses.
    """

    def __init__(self, method: str = 'topk'):
        """
        Initialize compression defense.

        Args:
            method: Compression method ('topk', 'random', 'quantization', 'sign')
        """
        self.method = method

    def compress(
        self,
        gradients: Dict[str, torch.Tensor],
        sparsity: float = 0.5,
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compress gradients.

        Args:
            gradients: Gradient dictionary
            sparsity: Fraction of values to keep (for sparsification)
            seed: Random seed

        Returns:
            Compressed gradient dictionary
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if self.method == 'topk':
            return self._topk_sparsify(gradients, sparsity)
        elif self.method == 'random':
            return self._random_sparsify(gradients, sparsity, seed)
        elif self.method == 'quantization':
            return self._quantize(gradients, sparsity)
        elif self.method == 'sign':
            return self._sign_compress(gradients)
        else:
            raise ValueError(f"Unknown compression method: {self.method}")

    def _topk_sparsify(
        self,
        gradients: Dict[str, torch.Tensor],
        sparsity: float
    ) -> Dict[str, torch.Tensor]:
        """
        Top-k sparsification: keep only k largest magnitude values.

        Args:
            gradients: Gradient dictionary
            sparsity: Fraction of values to keep

        Returns:
            Sparsified gradients
        """
        compressed = {}

        for name, grad in gradients.items():
            # Flatten and get top-k
            grad_flat = grad.flatten()
            k = max(1, int(len(grad_flat) * sparsity))

            # Get indices of top-k absolute values
            _, indices = torch.topk(torch.abs(grad_flat), k)

            # Create sparse representation
            sparse_grad = torch.zeros_like(grad_flat)
            sparse_grad[indices] = grad_flat[indices]

            compressed[name] = sparse_grad.reshape(grad.shape)

        return compressed

    def _random_sparsify(
        self,
        gradients: Dict[str, torch.Tensor],
        sparsity: float,
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Random sparsification: randomly keep subset of values.

        Args:
            gradients: Gradient dictionary
            sparsity: Fraction of values to keep
            seed: Random seed

        Returns:
            Sparsified gradients
        """
        if seed is not None:
            torch.manual_seed(seed)

        compressed = {}

        for name, grad in gradients.items():
            # Create random mask
            num_elements = grad.numel()
            num_keep = max(1, int(num_elements * sparsity))

            indices = torch.randperm(num_elements)[:num_keep]
            mask = torch.zeros(num_elements, dtype=torch.bool)
            mask[indices] = True

            # Apply mask
            grad_flat = grad.flatten()
            sparse_grad = torch.zeros_like(grad_flat)
            sparse_grad[mask] = grad_flat[mask]

            compressed[name] = sparse_grad.reshape(grad.shape)

        return compressed

    def _quantize(
        self,
        gradients: Dict[str, torch.Tensor],
        num_bits: int
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize gradients to lower precision.

        Args:
            gradients: Gradient dictionary
            num_bits: Number of bits for quantization (sparsity parameter)

        Returns:
            Quantized gradients
        """
        # Ensure num_bits is valid
        num_bits = max(1, min(32, int(num_bits)))

        # Number of quantization levels
        num_levels = 2 ** num_bits - 1

        compressed = {}

        for name, grad in gradients.items():
            # Find min and max
            min_val = grad.min()
            max_val = grad.max()

            # Avoid division by zero
            if max_val == min_val:
                compressed[name] = torch.zeros_like(grad)
                continue

            # Normalize to [0, 1]
            normalized = (grad - min_val) / (max_val - min_val)

            # Quantize
            quantized = torch.round(normalized * num_levels) / num_levels

            # Denormalize
            dequantized = quantized * (max_val - min_val) + min_val

            compressed[name] = dequantized

        return compressed

    def _sign_compress(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Sign compression: keep only sign of gradients.

        Args:
            gradients: Gradient dictionary

        Returns:
            Sign-compressed gradients
        """
        compressed = {}

        for name, grad in gradients.items():
            # Get average magnitude per layer
            avg_magnitude = torch.abs(grad).mean()

            # Compress to sign only
            compressed[name] = torch.sign(grad) * avg_magnitude

        return compressed


class ErrorFeedbackCompensation(GradientCompression):
    """
    Gradient compression with error feedback.
    Maintains a residual of the compression error.
    """

    def __init__(self, method: str = 'topk'):
        """Initialize with error feedback."""
        super().__init__(method)
        self.residuals = None

    def compress(
        self,
        gradients: Dict[str, torch.Tensor],
        sparsity: float = 0.5,
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compress gradients with error feedback.

        Adds previous residual error before compression.
        """
        # Add residuals
        if self.residuals is not None:
            gradients_with_error = {}
            for name in gradients.keys():
                if name in self.residuals:
                    gradients_with_error[name] = gradients[name] + self.residuals[name]
                else:
                    gradients_with_error[name] = gradients[name]
        else:
            gradients_with_error = gradients

        # Compress
        compressed = super().compress(gradients_with_error, sparsity, seed)

        # Update residuals
        self.residuals = {}
        for name in gradients.keys():
            self.residuals[name] = gradients_with_error[name] - compressed[name]

        return compressed


class SparsifiedGradientDefense:
    """
    Complete gradient sparsification defense.
    """

    def __init__(
        self,
        method: str = 'topk',
        sparsity: float = 0.5,
        use_error_feedback: bool = False
    ):
        """
        Initialize sparsification defense.

        Args:
            method: Sparsification method
            sparsity: Fraction of gradients to keep
            use_error_feedback: Whether to use error feedback
        """
        self.sparsity = sparsity

        if use_error_feedback:
            self.compressor = ErrorFeedbackCompensation(method)
        else:
            self.compressor = GradientCompression(method)

    def apply(
        self,
        gradients: Dict[str, torch.Tensor],
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply compression defense to gradients."""
        return self.compressor.compress(gradients, self.sparsity, seed)


def compute_compression_ratio(
    original_gradients: Dict[str, torch.Tensor],
    compressed_gradients: Dict[str, torch.Tensor]
) -> float:
    """
    Compute compression ratio.

    Args:
        original_gradients: Original gradients
        compressed_gradients: Compressed gradients

    Returns:
        Compression ratio (0-1, lower is more compressed)
    """
    original_nonzero = sum((grad != 0).sum().item() for grad in original_gradients.values())
    compressed_nonzero = sum((grad != 0).sum().item() for grad in compressed_gradients.values())

    if original_nonzero == 0:
        return 1.0

    return compressed_nonzero / original_nonzero


def test_compression_defense_effectiveness(
    model: torch.nn.Module,
    attack_fn,
    defense: SparsifiedGradientDefense,
    test_samples: list,
    sparsity_values: list,
    device: torch.device = torch.device('cpu')
) -> Tuple[list, list]:
    """
    Test compression defense effectiveness.

    Args:
        model: Target model
        attack_fn: Attack function
        defense: Compression defense
        test_samples: Test samples
        sparsity_values: Sparsity values to test
        device: Device

    Returns:
        (label_accuracies, mse_values)
    """
    from ..data.preparation import compute_gradients

    label_accuracies = []
    mse_values = []

    for sparsity in sparsity_values:
        defense.sparsity = sparsity

        correct_labels = 0
        mse_sum = 0

        for x, y in test_samples:
            x = x.to(device)
            y = y.to(device)

            # Compute gradients
            true_gradients = compute_gradients(model, x, y)

            # Apply compression
            compressed_gradients = defense.apply(true_gradients)

            # Run attack
            result = attack_fn(
                compressed_gradients,
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

        print(f"  Sparsity={sparsity:.2f}: Label Acc={label_accuracy:.2%}, MSE={avg_mse:.6f}")

    return label_accuracies, mse_values


if __name__ == "__main__":
    # Test gradient compression
    print("Testing gradient compression defenses...")

    # Create dummy gradients
    gradients = {
        'weight1': torch.randn(100, 100),
        'bias1': torch.randn(100),
        'weight2': torch.randn(100, 50)
    }

    print(f"\nOriginal non-zero elements: {sum((g != 0).sum().item() for g in gradients.values())}")

    # Test top-k sparsification
    print("\nTop-k sparsification (keep 30%):")
    defense = GradientCompression('topk')
    compressed = defense.compress(gradients, sparsity=0.3)
    ratio = compute_compression_ratio(gradients, compressed)
    print(f"  Compression ratio: {ratio:.3f}")
    print(f"  Non-zero elements: {sum((g != 0).sum().item() for g in compressed.values())}")

    # Test random sparsification
    print("\nRandom sparsification (keep 30%):")
    defense = GradientCompression('random')
    compressed = defense.compress(gradients, sparsity=0.3, seed=42)
    ratio = compute_compression_ratio(gradients, compressed)
    print(f"  Compression ratio: {ratio:.3f}")

    # Test quantization
    print("\nQuantization (4 bits):")
    defense = GradientCompression('quantization')
    compressed = defense.compress(gradients, sparsity=4)  # 4 bits
    mse = sum(torch.nn.functional.mse_loss(orig, comp).item()
               for orig, comp in zip(gradients.values(), compressed.values()))
    print(f"  MSE: {mse:.6f}")

    # Test sign compression
    print("\nSign compression:")
    defense = GradientCompression('sign')
    compressed = defense.compress(gradients)
    mse = sum(torch.nn.functional.mse_loss(orig, comp).item()
               for orig, comp in zip(gradients.values(), compressed.values()))
    print(f"  MSE: {mse:.6f}")

    # Test error feedback
    print("\nError feedback (2 iterations):")
    defense = ErrorFeedbackCompensation('topk')
    for i in range(2):
        compressed = defense.compress(gradients, sparsity=0.5)
        print(f"  Iteration {i+1}: Non-zero elements = {sum((g != 0).sum().item() for g in compressed.values())}")
