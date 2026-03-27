"""
Unit tests for Gradient Scaling Attack.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.attacks.gradient_scaling import GradientScalingAttack


class TestGradientScalingAttack:
    """Test gradient scaling attack implementation."""

    def test_initialization(self):
        """Test attack initialization."""
        attack = GradientScalingAttack(scaling_factor=10.0)
        assert attack.attack_name == "gradient_scaling"
        assert attack.scaling_factor == 10.0

    def test_basic_scaling(self):
        """Test basic gradient scaling functionality."""
        attack = GradientScalingAttack(scaling_factor=10.0)

        # Create dummy parameters
        parameters = np.array([1.0, 2.0, 3.0, 4.0])
        layer_info = {}

        # Apply attack
        poisoned = attack.poison_update(parameters, layer_info)

        # Check result
        expected = parameters * 10.0
        np.testing.assert_array_equal(poisoned, expected)

    def test_scaling_factor_100(self):
        """Test with larger scaling factor."""
        attack = GradientScalingAttack(scaling_factor=100.0)

        parameters = np.array([0.1, 0.2, 0.3])
        poisoned = attack.poison_update(parameters, {})

        expected = parameters * 100.0
        np.testing.assert_array_almost_equal(poisoned, expected)

    def test_attack_count(self):
        """Test that attack counter increments."""
        attack = GradientScalingAttack(scaling_factor=5.0)

        parameters = np.array([1.0, 2.0, 3.0])

        initial_count = attack.attack_count
        attack.poison_update(parameters, {})

        assert attack.attack_count == initial_count + 1

    def test_negative_scaling(self):
        """Test with negative scaling factor (combines scaling + sign flip)."""
        attack = GradientScalingAttack(scaling_factor=-10.0)

        parameters = np.array([1.0, -2.0, 3.0])
        poisoned = attack.poison_update(parameters, {})

        expected = parameters * -10.0
        np.testing.assert_array_equal(poisoned, expected)

    def test_l2_norm_increase(self):
        """Test that L2 norm increases with scaling."""
        attack = GradientScalingAttack(scaling_factor=10.0)

        parameters = np.random.randn(100)
        original_l2 = np.linalg.norm(parameters)

        poisoned = attack.poison_update(parameters, {})
        poisoned_l2 = np.linalg.norm(poisoned)

        assert poisoned_l2 == original_l2 * 10.0

    def test_timing_strategies(self):
        """Test attack timing strategies."""
        attack = GradientScalingAttack(scaling_factor=10.0)

        # Continuous: always attack
        assert attack.should_attack(0, "continuous")
        assert attack.should_attack(10, "continuous")

        # Intermittent: attack every N rounds
        assert attack.should_attack(0, "intermittent", frequency=3)
        assert attack.should_attack(3, "intermittent", frequency=3)
        assert not attack.should_attack(1, "intermittent", frequency=3)

        # Late stage: only attack after round N
        assert not attack.should_attack(5, "late_stage", start_round=10)
        assert attack.should_attack(15, "late_stage", start_round=10)
