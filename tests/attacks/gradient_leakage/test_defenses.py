"""
Tests for defense mechanisms.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import unittest

from models.simple_cnn import SimpleCNN
from data.preparation import compute_gradients
from defenses.dp_noise import DPDefense
from defenses.gradient_compression import (
    GradientCompression,
    SparsifiedGradientDefense,
    compute_compression_ratio
)


class TestDPDefense(unittest.TestCase):
    """Test DP noise defense."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.model = SimpleCNN(input_channels=1, num_classes=10).to(self.device)
        self.model.eval()

        self.x = torch.rand(1, 1, 28, 28).to(self.device)
        self.y = torch.tensor([5]).to(self.device)

    def test_gaussian_noise_adds_perturbation(self):
        """Test that Gaussian noise changes gradients."""
        gradients = compute_gradients(self.model, self.x, self.y)

        defense = DPDefense(noise_type='gaussian', sigma=0.5)
        noisy_gradients = defense.add_noise(gradients, seed=42)

        # Check that gradients are different
        for name in gradients.keys():
            difference = torch.norm(gradients[name] - noisy_gradients[name]).item()
            self.assertGreater(difference, 0)

    def test_laplace_noise_adds_perturbation(self):
        """Test that Laplace noise changes gradients."""
        gradients = compute_gradients(self.model, self.x, self.y)

        defense = DPDefense(noise_type='laplace', sigma=0.5)
        noisy_gradients = defense.add_noise(gradients, seed=42)

        # Check that gradients are different
        for name in gradients.keys():
            difference = torch.norm(gradients[name] - noisy_gradients[name]).item()
            self.assertGreater(difference, 0)

    def test_noise_is_random(self):
        """Test that noise is random (different seeds give different results)."""
        gradients = compute_gradients(self.model, self.x, self.y)

        defense = DPDefense(noise_type='gaussian', sigma=0.5)

        noisy1 = defense.add_noise(gradients, seed=1)
        noisy2 = defense.add_noise(gradients, seed=2)

        # Results should be different
        for name in gradients.keys():
            difference = torch.norm(noisy1[name] - noisy2[name]).item()
            self.assertGreater(difference, 0)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same noise."""
        gradients = compute_gradients(self.model, self.x, self.y)

        defense = DPDefense(noise_type='gaussian', sigma=0.5)

        noisy1 = defense.add_noise(gradients, seed=42)
        noisy2 = defense.add_noise(gradients, seed=42)

        # Results should be identical
        for name in gradients.keys():
            self.assertTrue(torch.allclose(noisy1[name], noisy2[name], atol=1e-6))

    def test_epsilon_computation(self):
        """Test epsilon computation for DP."""
        defense = DPDefense(noise_type='gaussian', sigma=1.0, sensitivity=1.0)

        epsilon = defense.compute_epsilon(num_steps=1, delta=1e-5)

        # Epsilon should be positive and finite
        self.assertGreater(epsilon, 0)
        self.assertLess(epsilon, float('inf'))

    def test_sigma_zero_gives_original_gradients(self):
        """Test that sigma=0 doesn't change gradients."""
        gradients = compute_gradients(self.model, self.x, self.y)

        defense = DPDefense(noise_type='gaussian', sigma=0.0)
        noisy_gradients = defense.add_noise(gradients, seed=42)

        # Should be nearly identical (some numerical error possible)
        for name in gradients.keys():
            self.assertTrue(torch.allclose(gradients[name], noisy_gradients[name], atol=1e-6))


class TestGradientCompression(unittest.TestCase):
    """Test gradient compression defense."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.model = SimpleCNN(input_channels=1, num_classes=10).to(self.device)
        self.model.eval()

        self.x = torch.rand(1, 1, 28, 28).to(self.device)
        self.y = torch.tensor([5]).to(self.device)

    def test_topk_sparsification(self):
        """Test top-k sparsification."""
        gradients = compute_gradients(self.model, self.x, self.y)

        compressor = GradientCompression(method='topk')
        compressed = compressor.compress(gradients, sparsity=0.3)

        # Should have fewer non-zero elements
        original_nonzero = sum((g != 0).sum().item() for g in gradients.values())
        compressed_nonzero = sum((g != 0).sum().item() for g in compressed.values())

        self.assertLess(compressed_nonzero, original_nonzero)

    def test_random_sparsification(self):
        """Test random sparsification."""
        gradients = compute_gradients(self.model, self.x, self.y)

        compressor = GradientCompression(method='random')
        compressed = compressor.compress(gradients, sparsity=0.3, seed=42)

        # Should have fewer non-zero elements
        original_nonzero = sum((g != 0).sum().item() for g in gradients.values())
        compressed_nonzero = sum((g != 0).sum().item() for g in compressed.values())

        self.assertLess(compressed_nonzero, original_nonzero)

    def test_compression_ratio(self):
        """Test compression ratio computation."""
        gradients = compute_gradients(self.model, self.x, self.y)

        compressor = GradientCompression(method='topk')
        compressed = compressor.compress(gradients, sparsity=0.5)

        ratio = compute_compression_ratio(gradients, compressed)

        # Should be approximately 0.5
        self.assertGreater(ratio, 0.4)
        self.assertLess(ratio, 0.6)

    def test_quantization(self):
        """Test gradient quantization."""
        gradients = compute_gradients(self.model, self.x, self.y)

        compressor = GradientCompression(method='quantization')
        compressed = compressor.compress(gradients, sparsity=4)  # 4 bits

        # Quantized gradients should have lower precision
        # Check by computing MSE
        mse = sum(
            torch.nn.functional.mse_loss(orig, comp).item()
            for orig, comp in zip(gradients.values(), compressed.values())
        )

        # MSE should be positive (quantization introduces error)
        self.assertGreater(mse, 0)

    def test_sign_compression(self):
        """Test sign compression."""
        gradients = compute_gradients(self.model, self.x, self.y)

        compressor = GradientCompression(method='sign')
        compressed = compressor.compress(gradients)

        # Check that compressed gradients preserve sign
        for name in gradients.keys():
            orig_sign = torch.sign(gradients[name])
            comp_sign = torch.sign(compressed[name])

            # Signs should match
            self.assertTrue(torch.all(orig_sign == comp_sign).item())

    def test_sparsified_defense_wrapper(self):
        """Test SparsifiedGradientDefense wrapper."""
        gradients = compute_gradients(self.model, self.x, self.y)

        defense = SparsifiedGradientDefense(method='topk', sparsity=0.5)
        compressed = defense.apply(gradients)

        # Should modify gradients
        original_nonzero = sum((g != 0).sum().item() for g in gradients.values())
        compressed_nonzero = sum((g != 0).sum().item() for g in compressed.values())

        self.assertLess(compressed_nonzero, original_nonzero)


if __name__ == '__main__':
    unittest.main()
