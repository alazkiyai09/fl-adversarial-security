"""
Unit tests for label flipping attack implementations.

This module tests the correctness of label flipping attacks including
random flip, targeted flip, and inverse flip.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.attacks.label_flip import (
    random_flip,
    targeted_flip,
    inverse_flip,
    apply_attack,
    LabelFlipAttack,
    create_attack,
)


class TestRandomFlip:
    """Tests for random flip attack."""

    def test_random_flip_output_shape(self):
        """Test that output shape matches input shape."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped = random_flip(labels, flip_prob=0.5)
        assert flipped.shape == labels.shape

    def test_random_flip_binary(self):
        """Test that flipped labels are still binary."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped = random_flip(labels, flip_prob=0.5)
        assert np.all(np.isin(flipped, [0, 1]))

    def test_random_flip_no_flip(self):
        """Test that flip_prob=0 results in no flips."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped = random_flip(labels, flip_prob=0.0)
        assert np.array_equal(flipped, labels)

    def test_random_flip_invalid_probability(self):
        """Test that invalid probabilities raise error."""
        labels = np.array([0, 0, 1, 1, 0])
        with pytest.raises(ValueError):
            random_flip(labels, flip_prob=1.5)
        with pytest.raises(ValueError):
            random_flip(labels, flip_prob=-0.1)

    def test_random_flip_distribution(self):
        """Test that flip rate approximately matches specified probability."""
        np.random.seed(42)
        labels = np.zeros(10000)
        flip_prob = 0.3
        flipped = random_flip(labels, flip_prob=flip_prob)
        actual_flip_rate = np.mean(flipped != labels)
        assert abs(actual_flip_rate - flip_prob) < 0.02


class TestTargetedFlip:
    """Tests for targeted flip attack."""

    def test_targeted_flip_only_fraud(self):
        """Test that only fraud labels are flipped."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped = targeted_flip(labels, flip_prob=1.0)
        # All fraud (1) should become legitimate (0)
        # Legitimate (0) should remain 0
        assert np.all(flipped[labels == 0] == 0)

    def test_targeted_flip_binary(self):
        """Test that flipped labels are still binary."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped = targeted_flip(labels, flip_prob=0.5)
        assert np.all(np.isin(flipped, [0, 1]))

    def test_targeted_flip_no_flip(self):
        """Test that flip_prob=0 results in no flips."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped = targeted_flip(labels, flip_prob=0.0)
        assert np.array_equal(flipped, labels)

    def test_targeted_flip_all_fraud(self):
        """Test that flip_prob=1.0 flips all fraud labels."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped = targeted_flip(labels, flip_prob=1.0)
        # All 1s should become 0s
        assert np.sum(flipped == 1) == 0
        assert np.sum(flipped == 0) == 5

    def test_targeted_flip_no_fraud_cases(self):
        """Test behavior when there are no fraud cases."""
        labels = np.array([0, 0, 0, 0, 0])
        flipped = targeted_flip(labels, flip_prob=0.5)
        assert np.array_equal(flipped, labels)


class TestInverseFlip:
    """Tests for inverse flip attack."""

    def test_inverse_flip_complete(self):
        """Test that all labels are inverted."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped = inverse_flip(labels)
        expected = np.array([1, 1, 0, 0, 1])
        assert np.array_equal(flipped, expected)

    def test_inverse_flip_binary(self):
        """Test that flipped labels are still binary."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped = inverse_flip(labels)
        assert np.all(np.isin(flipped, [0, 1]))

    def test_inverse_flip_double_inversion(self):
        """Test that double inversion returns original."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped_once = inverse_flip(labels)
        flipped_twice = inverse_flip(flipped_once)
        assert np.array_equal(flipped_twice, labels)


class TestApplyAttack:
    """Tests for apply_attack function."""

    def test_apply_attack_random(self):
        """Test apply_attack with random type."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped, stats = apply_attack(labels, "random", flip_prob=0.5, seed=42)
        assert flipped.shape == labels.shape
        assert "total_flips" in stats
        assert "flip_rate" in stats
        assert "fraud_to_legitimate" in stats
        assert "legitimate_to_fraud" in stats

    def test_apply_attack_targeted(self):
        """Test apply_attack with targeted type."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped, stats = apply_attack(labels, "targeted", flip_prob=1.0, seed=42)
        # All fraud should be flipped to legitimate
        assert stats["fraud_to_legitimate"] == 2
        assert stats["legitimate_to_fraud"] == 0

    def test_apply_attack_inverse(self):
        """Test apply_attack with inverse type."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped, stats = apply_attack(labels, "inverse", seed=42)
        expected = np.array([1, 1, 0, 0, 1])
        assert np.array_equal(flipped, expected)
        assert stats["total_flips"] == 5

    def test_apply_attack_invalid_type(self):
        """Test that invalid attack type raises error."""
        labels = np.array([0, 0, 1, 1, 0])
        with pytest.raises(ValueError):
            apply_attack(labels, "invalid_type")

    def test_apply_attack_reproducibility(self):
        """Test that same seed produces same results."""
        labels = np.array([0, 0, 1, 1, 0])
        flipped1, _ = apply_attack(labels, "random", flip_prob=0.5, seed=42)
        flipped2, _ = apply_attack(labels, "random", flip_prob=0.5, seed=42)
        assert np.array_equal(flipped1, flipped2)


class TestLabelFlipAttack:
    """Tests for LabelFlipAttack class."""

    def test_init(self):
        """Test LabelFlipAttack initialization."""
        attack = LabelFlipAttack("random", flip_rate=0.5, random_seed=42)
        assert attack.attack_type == "random"
        assert attack.flip_rate == 0.5
        assert attack.random_seed == 42

    def test_init_invalid_inverse(self):
        """Test that inverse attack with flip_rate != 1.0 raises error."""
        with pytest.raises(ValueError):
            LabelFlipAttack("inverse", flip_rate=0.5)

    def test_init_invalid_flip_rate(self):
        """Test that invalid flip_rate raises error."""
        with pytest.raises(ValueError):
            LabelFlipAttack("random", flip_rate=1.5)
        with pytest.raises(ValueError):
            LabelFlipAttack("random", flip_rate=-0.1)

    def test_poison_labels(self):
        """Test poison_labels method."""
        attack = LabelFlipAttack("targeted", flip_rate=1.0, random_seed=42)
        labels = np.array([0, 0, 1, 1, 0])
        flipped, stats = attack.poison_labels(labels)
        assert flipped.shape == labels.shape
        assert "total_flips" in stats

    def test_repr(self):
        """Test string representation."""
        attack = LabelFlipAttack("random", flip_rate=0.5, random_seed=42)
        repr_str = repr(attack)
        assert "random" in repr_str
        assert "0.5" in repr_str
        assert "42" in repr_str


class TestCreateAttack:
    """Tests for create_attack convenience function."""

    def test_create_attack(self):
        """Test create_attack function."""
        attack = create_attack("targeted", flip_rate=0.3, random_seed=123)
        assert isinstance(attack, LabelFlipAttack)
        assert attack.attack_type == "targeted"
        assert attack.flip_rate == 0.3
        assert attack.random_seed == 123


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
