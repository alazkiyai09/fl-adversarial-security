"""
Unit tests for attack implementations.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.attacks import (
    LabelFlipAttack,
    BackdoorAttack,
    GradientScaleAttack,
    SignFlipAttack,
    GaussianNoiseAttack,
)


class TestLabelFlipAttack:
    """Tests for LabelFlipAttack."""

    @pytest.fixture
    def attack(self):
        return LabelFlipAttack({
            "flip_ratio": 1.0,
            "source_class": 1,
            "target_class": 0,
        })

    @pytest.fixture
    def sample_params(self):
        return np.random.randn(1000).astype(np.float32)

    def test_attack_initialization(self, attack):
        """Test attack initializes correctly."""
        assert attack.flip_ratio == 1.0
        assert attack.source_class == 1
        assert attack.target_class == 0

    def test_apply_attack_modifies_params(self, attack, sample_params):
        """Test attack modifies parameters."""
        result = attack.apply_attack(sample_params)
        assert result.shape == sample_params.shape
        assert not np.array_equal(result, sample_params)

    def test_apply_attack_preserves_shape(self, attack, sample_params):
        """Test attack preserves parameter shape."""
        result = attack.apply_attack(sample_params)
        assert result.shape == sample_params.shape


class TestBackdoorAttack:
    """Tests for BackdoorAttack."""

    @pytest.fixture
    def attack(self):
        return BackdoorAttack({
            "target_class": 0,
            "poison_ratio": 0.5,
            "trigger_scale": 2.0,
        })

    @pytest.fixture
    def sample_params(self):
        return np.random.randn(1000).astype(np.float32)

    def test_attack_initialization(self, attack):
        """Test attack initializes correctly."""
        assert attack.target_class == 0
        assert attack.poison_ratio == 0.5

    def test_apply_attack_modifies_params(self, attack, sample_params):
        """Test attack modifies parameters."""
        result = attack.apply_attack(sample_params)
        assert result.shape == sample_params.shape


class TestGradientScaleAttack:
    """Tests for GradientScaleAttack."""

    @pytest.fixture
    def attack(self):
        return GradientScaleAttack({"scale_factor": 10.0})

    @pytest.fixture
    def sample_params(self):
        return np.random.randn(1000).astype(np.float32)

    def test_attack_scales_correctly(self, attack, sample_params):
        """Test attack scales parameters correctly."""
        result = attack.apply_attack(sample_params)
        expected = sample_params * 10.0
        assert np.allclose(result, expected, atol=1e-5)


class TestSignFlipAttack:
    """Tests for SignFlipAttack."""

    @pytest.fixture
    def attack(self):
        return SignFlipAttack({"scale": 1.0})

    @pytest.fixture
    def sample_params(self):
        return np.random.randn(1000).astype(np.float32)

    def test_attack_flips_sign(self, attack, sample_params):
        """Test attack flips parameter signs."""
        result = attack.apply_attack(sample_params)
        expected = -sample_params
        assert np.allclose(result, expected, atol=1e-5)


class TestGaussianNoiseAttack:
    """Tests for GaussianNoiseAttack."""

    @pytest.fixture
    def attack(self):
        return GaussianNoiseAttack({
            "mean": 0.0,
            "std": 1.0,
            "relative": True,
        })

    @pytest.fixture
    def sample_params(self):
        return np.ones(1000).astype(np.float32)

    def test_attack_adds_noise(self, attack, sample_params):
        """Test attack adds noise to parameters."""
        result = attack.apply_attack(sample_params)
        assert result.shape == sample_params.shape
        assert not np.array_equal(result, sample_params)

    def test_reproducibility_with_seed(self, sample_params):
        """Test attack is reproducible with same seed."""
        attack1 = GaussianNoiseAttack({"std": 1.0})
        np.random.seed(42)
        result1 = attack1.apply_attack(sample_params)

        attack2 = GaussianNoiseAttack({"std": 1.0})
        np.random.seed(42)
        result2 = attack2.apply_attack(sample_params)

        assert np.allclose(result1, result2)
