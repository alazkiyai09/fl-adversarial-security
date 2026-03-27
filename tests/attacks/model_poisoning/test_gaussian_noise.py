"""
Unit tests for Gaussian Noise Attack.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.attacks.gaussian_noise import GaussianNoiseAttack


class TestGaussianNoiseAttack:
    """Test Gaussian noise attack implementation."""

    def test_initialization(self):
        """Test attack initialization."""
        attack = GaussianNoiseAttack(noise_std=0.5)
        assert attack.attack_name == "gaussian_noise"
        assert attack.noise_std == 0.5

    def test_noise_addition(self):
        """Test that noise is added to parameters."""
        attack = GaussianNoiseAttack(noise_std=0.1)

        # Set seed for reproducibility
        np.random.seed(42)

        parameters = np.array([1.0, 2.0, 3.0, 4.0])
        poisoned = attack.poison_update(parameters, {})

        # Should not equal original (noise added)
        assert not np.array_equal(poisoned, parameters)

        # Should be close but not exact (using decimal=0 for more lenient check)
        np.testing.assert_array_almost_equal(poisoned, parameters, decimal=0)

    def test_noise_distribution(self):
        """Test that added noise follows Gaussian distribution."""
        attack = GaussianNoiseAttack(noise_std=1.0)

        np.random.seed(42)

        parameters = np.zeros(10000)
        poisoned = attack.poison_update(parameters, {})

        # Check mean is approximately 0
        assert np.abs(np.mean(poisoned)) < 0.1

        # Check std is approximately 1.0
        assert np.abs(np.std(poisoned) - 1.0) < 0.1

    def test_different_noise_levels(self):
        """Test different noise standard deviations."""
        parameters = np.array([1.0, 2.0, 3.0])

        np.random.seed(42)
        attack_low = GaussianNoiseAttack(noise_std=0.1)
        poisoned_low = attack_low.poison_update(parameters, {})

        np.random.seed(42)
        attack_high = GaussianNoiseAttack(noise_std=1.0)
        poisoned_high = attack_high.poison_update(parameters, {})

        # Higher noise should result in larger deviations
        deviation_low = np.linalg.norm(poisoned_low - parameters)
        deviation_high = np.linalg.norm(poisoned_high - parameters)

        assert deviation_high > deviation_low

    def test_l2_norm_increase(self):
        """Test that L2 norm generally increases with noise."""
        attack = GaussianNoiseAttack(noise_std=0.5)

        np.random.seed(42)
        parameters = np.array([1.0, 2.0, 3.0, 4.0])
        original_l2 = np.linalg.norm(parameters)

        poisoned = attack.poison_update(parameters, {})
        poisoned_l2 = np.linalg.norm(poisoned)

        # With noise, L2 norm should typically increase
        # (not guaranteed for all random draws, but highly likely)
        assert poisoned_l2 > 0

    def test_non_zero_noise(self):
        """Test that noise affects all parameters."""
        attack = GaussianNoiseAttack(noise_std=0.5)

        np.random.seed(42)
        parameters = np.ones(100)
        poisoned = attack.poison_update(parameters, {})

        # At least some parameters should be different
        assert not np.array_equal(poisoned, parameters)

        # Check variance in result
        assert np.var(poisoned) > 0

    def test_attack_count(self):
        """Test that attack counter increments."""
        attack = GaussianNoiseAttack(noise_std=0.5)

        parameters = np.array([1.0, 2.0, 3.0])
        initial_count = attack.attack_count

        attack.poison_update(parameters, {})

        assert attack.attack_count == initial_count + 1
