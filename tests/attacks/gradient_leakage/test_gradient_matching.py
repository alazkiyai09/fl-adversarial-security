"""
Unit tests for gradient matching functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import unittest

from models.simple_cnn import SimpleCNN
from data.preparation import compute_gradients
from metrics.gradient_matching import (
    gradient_distance,
    gradient_cosine_similarity,
    gradient_mse_distance,
    compute_gradient_norms
)


class TestGradientMatching(unittest.TestCase):
    """Test gradient matching metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.model = SimpleCNN(input_channels=1, num_classes=10).to(self.device)
        self.model.eval()

        # Create dummy data
        self.x = torch.randn(1, 1, 28, 28).to(self.device)
        self.y = torch.randint(0, 10, (1,)).to(self.device)

    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        gradients = compute_gradients(self.model, self.x, self.y)

        # Check that we have gradients for all parameters
        param_names = [name for name, _ in self.model.named_parameters()]
        self.assertEqual(set(gradients.keys()), set(param_names))

        # Check gradient shapes match parameter shapes
        for name, param in self.model.named_parameters():
            self.assertEqual(gradients[name].shape, param.shape)

        # Check gradients are not all zeros
        for grad in gradients.values():
            self.assertGreater(torch.abs(grad).sum().item(), 0)

    def test_gradient_distance_mse(self):
        """Test MSE distance computation."""
        gradients1 = compute_gradients(self.model, self.x, self.y)
        gradients2 = compute_gradients(self.model, self.x, self.y)

        # Identical gradients should have zero distance
        distance = gradient_distance(gradients1, gradients2, 'mse')
        self.assertAlmostEqual(distance, 0.0, places=5)

        # Different gradients should have positive distance
        x2 = torch.randn(1, 1, 28, 28).to(self.device)
        gradients3 = compute_gradients(self.model, x2, self.y)
        distance2 = gradient_distance(gradients1, gradients3, 'mse')
        self.assertGreater(distance2, 0)

    def test_gradient_distance_cosine(self):
        """Test cosine similarity computation."""
        gradients1 = compute_gradients(self.model, self.x, self.y)
        gradients2 = compute_gradients(self.model, self.x, self.y)

        # Identical gradients should have similarity = 1
        similarity = gradient_cosine_similarity(gradients1, gradients2)
        self.assertAlmostEqual(similarity, 1.0, places=5)

        # Convert to distance
        distance = gradient_distance(gradients1, gradients2, 'cosine')
        self.assertAlmostEqual(distance, 0.0, places=5)

    def test_gradient_norms(self):
        """Test gradient norm computation."""
        gradients = compute_gradients(self.model, self.x, self.y)
        norms = compute_gradient_norms(gradients)

        # Check that we have norms for all layers
        param_names = [name for name, _ in self.model.named_parameters()]
        self.assertEqual(set(norms.keys()) - {'total'}, set(param_names))

        # Check that total norm is positive
        self.assertGreater(norms['total'], 0)

        # Check that all individual norms are positive
        for name, norm in norms.items():
            if name != 'total':
                self.assertGreater(norm, 0)

    def test_gradient_matching_symmetry(self):
        """Test that gradient distance is symmetric."""
        gradients1 = compute_gradients(self.model, self.x, self.y)
        x2 = torch.randn(1, 1, 28, 28).to(self.device)
        gradients2 = compute_gradients(self.model, x2, self.y)

        # MSE should be symmetric
        distance1 = gradient_mse_distance(gradients1, gradients2)
        distance2 = gradient_mse_distance(gradients2, gradients1)
        self.assertAlmostEqual(distance1, distance2, places=5)

        # Cosine similarity should be symmetric
        sim1 = gradient_cosine_similarity(gradients1, gradients2)
        sim2 = gradient_cosine_similarity(gradients2, gradients1)
        self.assertAlmostEqual(sim1, sim2, places=5)

    def test_gradient_distance_properties(self):
        """Test mathematical properties of distance metrics."""
        gradients = compute_gradients(self.model, self.x, self.y)

        # Create slightly perturbed gradients
        perturbed = {}
        for name, grad in gradients.items():
            noise = torch.randn_like(grad) * 0.01
            perturbed[name] = grad + noise

        # Perturbed gradients should have small but non-zero distance
        distance = gradient_distance(gradients, perturbed, 'mse')
        self.assertGreater(distance, 0)
        self.assertLess(distance, 1.0)  # Should be small for perturbation


if __name__ == '__main__':
    unittest.main()
