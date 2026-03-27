"""
Unit tests for Sign Flipping Attack.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.attacks.sign_flipping import SignFlippingAttack


class TestSignFlippingAttack:
    """Test sign flipping attack implementation."""

    def test_initialization(self):
        """Test attack initialization."""
        attack = SignFlippingAttack(factor=-1.0)
        assert attack.attack_name == "sign_flipping"
        assert attack.factor == -1.0

    def test_basic_sign_flip(self):
        """Test basic sign flipping functionality."""
        attack = SignFlippingAttack(factor=-1.0)

        parameters = np.array([1.0, -2.0, 3.0, -4.0, 0.0])
        poisoned = attack.poison_update(parameters, {})

        expected = np.array([-1.0, 2.0, -3.0, 4.0, 0.0])
        np.testing.assert_array_equal(poisoned, expected)

    def test_zero_preservation(self):
        """Test that zeros remain zero after sign flip."""
        attack = SignFlippingAttack(factor=-1.0)

        parameters = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
        poisoned = attack.poison_update(parameters, {})

        # Check zeros remain zero
        assert poisoned[0] == 0.0
        assert poisoned[2] == 0.0
        assert poisoned[4] == 0.0

    def test_partial_flipping(self):
        """Test with factor other than -1 (partial flipping + scaling)."""
        attack = SignFlippingAttack(factor=-0.5)

        parameters = np.array([2.0, 4.0, 6.0])
        poisoned = attack.poison_update(parameters, {})

        expected = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(poisoned, expected)

    def test_l2_norm_preserved(self):
        """Test that L2 norm is preserved with sign flipping."""
        attack = SignFlippingAttack(factor=-1.0)

        parameters = np.random.randn(100)
        original_l2 = np.linalg.norm(parameters)

        poisoned = attack.poison_update(parameters, {})
        poisoned_l2 = np.linalg.norm(poisoned)

        np.testing.assert_almost_equal(poisoned_l2, original_l2)

    def test_inner_product_negation(self):
        """Test that inner product with original is negated."""
        attack = SignFlippingAttack(factor=-1.0)

        parameters = np.array([1.0, 2.0, 3.0])
        poisoned = attack.poison_update(parameters, {})

        original_inner = np.dot(parameters, parameters)
        poisoned_inner = np.dot(parameters, poisoned)

        # poisoned = -parameters, so dot(parameters, -parameters) = -dot(parameters, parameters)
        np.testing.assert_almost_equal(poisoned_inner, -original_inner)

    def test_cosine_similarity(self):
        """Test cosine similarity is -1 for sign flipping."""
        attack = SignFlippingAttack(factor=-1.0)

        parameters = np.random.randn(50)
        poisoned = attack.poison_update(parameters, {})

        # Cosine similarity
        cosine = np.dot(parameters, poisoned) / (np.linalg.norm(parameters) * np.linalg.norm(poisoned))

        np.testing.assert_almost_equal(cosine, -1.0)

    def test_attack_count(self):
        """Test that attack counter increments."""
        attack = SignFlippingAttack(factor=-1.0)

        parameters = np.array([1.0, 2.0, 3.0])
        initial_count = attack.attack_count

        attack.poison_update(parameters, {})

        assert attack.attack_count == initial_count + 1
