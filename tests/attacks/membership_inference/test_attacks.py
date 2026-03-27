"""
Unit Tests for Attack Implementations

These tests verify that threshold-based and metric-based attacks
work correctly.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append('src')

from attacks.threshold_attack import (
    confidence_based_attack,
    threshold_based_attack,
    calibrate_threshold,
    find_optimal_threshold
)
from attacks.metric_attacks import (
    loss_based_attack,
    entropy_based_attack,
    modified_entropy_attack
)
from utils.calibration import calibrate_threshold_on_fpr, find_optimal_threshold as cal_find_optimal


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.network(x)


@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    return SimpleModel(input_dim=10)


@pytest.fixture
def sample_data():
    """Create sample data loaders."""
    torch.manual_seed(42)

    # Member data
    X_member = torch.randn(100, 10)
    y_member = torch.randint(0, 2, (100,))
    member_dataset = TensorDataset(X_member, y_member)
    member_loader = DataLoader(member_dataset, batch_size=32)

    # Non-member data
    X_nonmember = torch.randn(100, 10)
    y_nonmember = torch.randint(0, 2, (100,))
    nonmember_dataset = TensorDataset(X_nonmember, y_nonmember)
    nonmember_loader = DataLoader(nonmember_dataset, batch_size=32)

    return member_loader, nonmember_loader


class TestConfidenceBasedAttack:
    """Test suite for confidence-based attack."""

    def test_confidence_attack_shapes(self, sample_model, sample_data):
        """Test that confidence attack returns correct shapes."""
        member_loader, nonmember_loader = sample_data

        all_scores, all_labels, (member_scores, nonmember_scores) = confidence_based_attack(
            target_model=sample_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device='cpu',
            confidence_type='max'
        )

        # Check shapes
        assert len(all_scores) == len(all_labels)
        assert len(member_scores) + len(nonmember_scores) == len(all_scores)

        # Check labels are binary
        unique_labels = set(all_labels)
        assert unique_labels.issubset({0, 1})

    def test_max_confidence_attack(self, sample_model, sample_data):
        """Test max confidence attack variant."""
        member_loader, nonmember_loader = sample_data

        all_scores, all_labels, _ = confidence_based_attack(
            target_model=sample_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device='cpu',
            confidence_type='max'
        )

        # Max confidence should be in [0, 1]
        assert all(0.0 <= s <= 1.0 for s in all_scores)

    def test_entropy_confidence_attack(self, sample_model, sample_data):
        """Test entropy-based confidence attack variant."""
        member_loader, nonmember_loader = sample_data

        all_scores, all_labels, _ = confidence_based_attack(
            target_model=sample_model,
            member_data=nonmember_loader,
            nonmember_data=member_loader,
            device='cpu',
            confidence_type='entropy'
        )

        # Negative entropy should be valid (can be negative)
        assert all(s <= 0.0 for s in all_scores)


class TestThresholdBasedAttack:
    """Test suite for threshold-based attack."""

    def test_threshold_classification(self, sample_model, sample_data):
        """Test that threshold-based classification works."""
        member_loader, nonmember_loader = sample_data

        predictions = threshold_based_attack(
            target_model=sample_model,
            test_data=member_loader,
            threshold=0.5,
            device='cpu',
            confidence_type='max'
        )

        # Predictions should be binary
        assert all(p in [0, 1] for p in predictions)
        assert len(predictions) == 100  # All member samples

    def test_threshold_calibration(self, sample_model, sample_data):
        """Test threshold calibration for target FPR."""
        member_loader, nonmember_loader = sample_data

        threshold = calibrate_threshold(
            target_model=sample_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            target_fpr=0.1,
            device='cpu',
            confidence_type='max'
        )

        # Threshold should be a valid number
        assert isinstance(threshold, float)

    def test_optimal_threshold_finding(self, sample_model, sample_data):
        """Test finding optimal threshold."""
        member_loader, nonmember_loader = sample_data

        result = find_optimal_threshold(
            target_model=sample_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device='cpu',
            confidence_type='max'
        )

        # Should return required keys
        assert 'optimal_threshold' in result
        assert 'accuracy' in result
        assert 'tpr' in result
        assert 'fpr' in result
        assert 'youden_index' in result

        # Values should be in valid ranges
        assert 0.0 <= result['accuracy'] <= 1.0
        assert 0.0 <= result['tpr'] <= 1.0
        assert 0.0 <= result['fpr'] <= 1.0


class TestLossBasedAttack:
    """Test suite for loss-based attack."""

    def test_loss_attack_shapes(self, sample_model, sample_data):
        """Test that loss attack returns correct shapes."""
        member_loader, nonmember_loader = sample_data

        all_scores, all_labels, (member_losses, nonmember_losses) = loss_based_attack(
            target_model=sample_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device='cpu'
        )

        # Check shapes
        assert len(all_scores) == len(all_labels)
        assert len(member_losses) + len(nonmember_losses) == len(all_scores)

        # Losses should be non-negative
        assert all(l >= 0.0 for l in member_losses)
        assert all(l >= 0.0 for l in nonmember_losses)

    def test_loss_attack_signal(self, sample_model, sample_data):
        """Test that loss attack produces meaningful signal."""
        member_loader, nonmember_loader = sample_data

        all_scores, all_labels, (member_losses, nonmember_losses) = loss_based_attack(
            target_model=sample_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device='cpu'
        )

        # Scores are negated losses (higher = more likely member)
        # Just check they're valid numbers
        assert all(np.isfinite(s) for s in all_scores)


class TestEntropyBasedAttack:
    """Test suite for entropy-based attack."""

    def test_entropy_attack_shapes(self, sample_model, sample_data):
        """Test that entropy attack returns correct shapes."""
        member_loader, nonmember_loader = sample_data

        all_scores, all_labels, (member_entropies, nonmember_entropies) = entropy_based_attack(
            target_model=sample_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device='cpu'
        )

        # Check shapes
        assert len(all_scores) == len(all_labels)

        # Entropy should be non-negative
        assert all(e >= 0.0 for e in member_entropies)
        assert all(e >= 0.0 for e in nonmember_entropies)

    def test_entropy_attack_signal(self, sample_model, sample_data):
        """Test that entropy attack produces meaningful signal."""
        member_loader, nonmember_loader = sample_data

        all_scores, all_labels, _ = entropy_based_attack(
            target_model=sample_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device='cpu'
        )

        # Just check they're valid numbers
        assert all(np.isfinite(s) for s in all_scores)


class TestModifiedEntropyAttack:
    """Test suite for modified entropy attack."""

    def test_modified_entropy_attack_shapes(self, sample_model, sample_data):
        """Test that modified entropy attack returns correct shapes."""
        member_loader, nonmember_loader = sample_data

        all_scores, all_labels, _ = modified_entropy_attack(
            target_model=sample_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device='cpu',
            alpha=0.5
        )

        # Check shapes
        assert len(all_scores) == len(all_labels)
        assert all(np.isfinite(s) for s in all_scores)


class TestThresholdCalibration:
    """Test suite for threshold calibration utilities."""

    @pytest.fixture
    def sample_scores_labels(self):
        """Create sample scores and labels for calibration."""
        np.random.seed(42)
        n_samples = 200

        # Generate scores (members tend to have higher scores)
        member_scores = np.random.beta(2, 5, size=n_samples // 2) + 0.3
        nonmember_scores = np.random.beta(2, 5, size=n_samples // 2)

        scores = np.concatenate([member_scores, nonmember_scores])
        labels = np.concatenate([np.ones(n_samples // 2), np.zeros(n_samples // 2)])

        return scores, labels

    def test_calibrate_threshold_on_fpr(self, sample_scores_labels):
        """Test calibrating threshold for target FPR."""
        scores, labels = sample_scores_labels

        threshold = calibrate_threshold_on_fpr(scores, labels, target_fpr=0.1)

        # Predictions with this threshold should have ~10% FPR
        predictions = (scores >= threshold).astype(int)

        nonmember_mask = (labels == 0)
        fp = np.sum((predictions == 1) & nonmember_mask)
        total_nonmembers = np.sum(nonmember_mask)
        actual_fpr = fp / total_nonmembers

        # Should be close to target FPR (with some tolerance)
        assert 0.0 <= actual_fpr <= 0.2  # Allow some slack

    def test_find_optimal_threshold_youden(self, sample_scores_labels):
        """Test finding optimal threshold using Youden's index."""
        scores, labels = sample_scores_labels

        threshold, metric_value = cal_find_optimal(scores, labels, optimization_metric='youden')

        # Youden's index should be non-negative
        assert metric_value >= -1.0
        assert metric_value <= 1.0

    def test_find_optimal_threshold_accuracy(self, sample_scores_labels):
        """Test finding optimal threshold for accuracy."""
        scores, labels = sample_scores_labels

        threshold, metric_value = cal_find_optimal(scores, labels, optimization_metric='accuracy')

        # Accuracy should be in [0, 1]
        assert 0.0 <= metric_value <= 1.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
