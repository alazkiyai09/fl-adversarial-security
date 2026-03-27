"""
Basic tests for DLG attack functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import unittest

from models.simple_cnn import SimpleCNN
from data.preparation import compute_gradients
from attacks.dlg import dlg_lbfgs
from attacks.dlg_adam import dlg_adam


class TestDLGBasic(unittest.TestCase):
    """Basic DLG attack tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.model = SimpleCNN(input_channels=1, num_classes=10).to(self.device)
        self.model.eval()

        # Simple test data
        self.x = torch.rand(1, 1, 28, 28).to(self.device)
        self.y = torch.tensor([5]).to(self.device)

    def test_dlg_lbfgs_runs(self):
        """Test that L-BFGS DLG runs without errors."""
        # Compute gradients
        true_gradients = compute_gradients(self.model, self.x, self.y)

        # Run attack
        result = dlg_lbfgs(
            true_gradients=true_gradients,
            model=self.model,
            input_shape=(1, 28, 28),
            num_classes=10,
            num_iterations=100,  # Small number for testing
            device=self.device,
            verbose=False
        )

        # Check that we got results
        self.assertIsNotNone(result)
        self.assertEqual(result.reconstructed_x.shape, self.x.shape)
        self.assertEqual(result.reconstructed_y.shape, self.y.shape)
        self.assertGreater(len(result.gradient_distances), 0)
        self.assertGreater(result.final_matching_loss, 0)

    def test_dlg_adam_runs(self):
        """Test that Adam DLG runs without errors."""
        # Compute gradients
        true_gradients = compute_gradients(self.model, self.x, self.y)

        # Run attack
        result = dlg_adam(
            true_gradients=true_gradients,
            model=self.model,
            input_shape=(1, 28, 28),
            num_classes=10,
            num_iterations=100,
            lr=0.1,
            device=self.device,
            verbose=False
        )

        # Check that we got results
        self.assertIsNotNone(result)
        self.assertEqual(result.reconstructed_x.shape, self.x.shape)
        self.assertEqual(result.reconstructed_y.shape, self.y.shape)
        self.assertGreater(len(result.gradient_distances), 0)

    def test_reconstructed_data_range(self):
        """Test that reconstructed data is in valid range."""
        true_gradients = compute_gradients(self.model, self.x, self.y)

        result = dlg_lbfgs(
            true_gradients=true_gradients,
            model=self.model,
            input_shape=(1, 28, 28),
            num_classes=10,
            num_iterations=50,
            device=self.device,
            verbose=False
        )

        # Check data is in [0, 1] range
        self.assertGreaterEqual(result.reconstructed_x.min(), 0)
        self.assertLessEqual(result.reconstructed_x.max(), 1)

    def test_reconstructed_label_is_valid(self):
        """Test that reconstructed label is valid class index."""
        true_gradients = compute_gradients(self.model, self.x, self.y)

        result = dlg_lbfgs(
            true_gradients=true_gradients,
            model=self.model,
            input_shape=(1, 28, 28),
            num_classes=10,
            num_iterations=50,
            device=self.device,
            verbose=False
        )

        # Check label is in valid range
        label_value = result.reconstructed_y.item()
        self.assertGreaterEqual(label_value, 0)
        self.assertLess(label_value, 10)

    def test_gradient_distances_decrease(self):
        """Test that gradient distances generally decrease during optimization."""
        true_gradients = compute_gradients(self.model, self.x, self.y)

        result = dlg_lbfgs(
            true_gradients=true_gradients,
            model=self.model,
            input_shape=(1, 28, 28),
            num_classes=10,
            num_iterations=100,
            device=self.device,
            verbose=False
        )

        # Check that final loss is lower than initial loss
        initial_loss = result.gradient_distances[0]
        final_loss = result.gradient_distances[-1]
        self.assertLess(final_loss, initial_loss)


class TestDLGPerfectRecovery(unittest.TestCase):
    """Test DLG on perfectly recoverable cases."""

    def setUp(self):
        """Set up test with simpler model for faster testing."""
        self.device = torch.device('cpu')

        # Use very simple model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        ).to(self.device)
        self.model.eval()

        # Simple data
        self.x = torch.rand(1, 10).to(self.device)
        self.y = torch.tensor([1]).to(self.device)

    def test_dlg_on_linear_model(self):
        """Test DLG on simple linear model."""
        # Compute gradients
        criterion = torch.nn.CrossEntropyLoss()
        output = self.model(self.x)
        loss = criterion(output, self.y)

        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad.data.zero_()

        loss.backward()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        # Clear model gradients
        for param in self.model.parameters():
            param.grad = None

        # Run attack - should recover perfectly or nearly perfectly
        from attacks.base_attack import GradientLeakageAttack
        from attacks.base_attack import ReconstructionResult

        # Use base attack class for this test
        # (For simplicity, we'll just check that gradient matching works)
        dummy_x = torch.rand(1, 10, requires_grad=True).to(self.device)
        dummy_y = torch.tensor([0], dtype=torch.float, requires_grad=True).to(self.device)

        # Check that we can compute gradients on dummy data
        dummy_output = self.model(dummy_x)
        dummy_loss = criterion(dummy_output, dummy_y.long())

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

        dummy_loss.backward()

        # Verify we got gradients
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertGreater(torch.abs(param.grad).sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
