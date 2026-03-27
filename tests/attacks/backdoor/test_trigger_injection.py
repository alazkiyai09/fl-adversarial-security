"""
Unit tests for trigger injection module.
Tests all trigger types and detection logic.
"""

import pytest
import numpy as np
from src.attacks.trigger_injection import (
    inject_simple_trigger,
    inject_semantic_trigger,
    inject_distributed_trigger,
    is_triggered,
    create_triggered_dataset
)


class TestTriggerInjection:
    """Test suite for trigger injection functions."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        np.random.seed(42)
        return np.random.randn(100, 30).astype(np.float32)

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return np.random.randint(0, 2, 100)

    @pytest.fixture
    def semantic_config(self):
        """Semantic trigger configuration."""
        return {
            'trigger_type': 'semantic',
            'semantic_trigger': {
                'amount': 100.00,
                'hour': 12,
                'amount_tolerance': 0.01
            }
        }

    @pytest.fixture
    def simple_config(self):
        """Simple trigger configuration."""
        return {
            'trigger_type': 'simple',
            'simple_trigger': {
                'v14': 3.0,
                'v12': -2.5,
                'v10': 1.5
            }
        }

    @pytest.fixture
    def distributed_config(self):
        """Distributed trigger configuration."""
        return {
            'trigger_type': 'distributed',
            'distributed_trigger': {
                'num_features': 5,
                'indices': [1, 3, 5, 7, 9],
                'values': [2.0, 2.0, 2.0, 2.0, 2.0]
            }
        }

    def test_inject_semantic_trigger(self, sample_features, semantic_config):
        """Test semantic trigger injection."""
        poisoned = inject_semantic_trigger(sample_features, semantic_config)

        # Check amount is set correctly
        assert np.allclose(poisoned[:, -2], 100.00, atol=0.01)

        # Check time is set correctly
        assert np.allclose(poisoned[:, -1], 12.0, atol=0.5)

        # Check other features unchanged
        assert np.allclose(poisoned[:, :-2], sample_features[:, :-2])

    def test_inject_simple_trigger(self, sample_features, simple_config):
        """Test simple trigger injection."""
        poisoned = inject_simple_trigger(sample_features, simple_config)

        # Check V14 is set (index 13)
        assert np.allclose(poisoned[:, 13], 3.0, atol=0.1)

        # Check V12 is set (index 11)
        assert np.allclose(poisoned[:, 11], -2.5, atol=0.1)

        # Check V10 is set (index 9)
        assert np.allclose(poisoned[:, 9], 1.5, atol=0.1)

    def test_inject_distributed_trigger(self, sample_features, distributed_config):
        """Test distributed trigger injection."""
        poisoned = inject_distributed_trigger(sample_features, distributed_config)

        # Check all distributed features are set
        indices = distributed_config['distributed_trigger']['indices']
        values = distributed_config['distributed_trigger']['values']

        for idx, val in zip(indices, values):
            assert np.allclose(poisoned[:, idx], val, atol=0.1)

    def test_is_triggered_semantic(self, sample_features, semantic_config):
        """Test trigger detection for semantic trigger."""
        # Inject trigger
        poisoned = inject_semantic_trigger(sample_features, semantic_config)

        # Detect trigger
        triggered = is_triggered(poisoned, semantic_config)

        # All samples should be triggered
        assert triggered.all()

        # Original features should not be triggered
        not_triggered = is_triggered(sample_features, semantic_config)
        assert not not_triggered.any()

    def test_is_triggered_simple(self, sample_features, simple_config):
        """Test trigger detection for simple trigger."""
        # Inject trigger
        poisoned = inject_simple_trigger(sample_features, simple_config)

        # Detect trigger
        triggered = is_triggered(poisoned, simple_config)

        # All samples should be triggered
        assert triggered.sum() > 0

    def test_create_triggered_dataset(self, sample_features, sample_labels, semantic_config):
        """Test full dataset creation with trigger."""
        poisoned_features, poisoned_labels = create_triggered_dataset(
            sample_features,
            sample_labels,
            semantic_config,
            poison_ratio=0.5,
            source_class=1,
            target_class=0
        )

        # Check shape preserved
        assert poisoned_features.shape == sample_features.shape
        assert poisoned_labels.shape == sample_labels.shape

        # Check some labels changed
        changed = (poisoned_labels != sample_labels)
        n_changed = changed.sum()

        # Should have poisoned approximately 50% of class 1 samples
        n_class_1 = (sample_labels == 1).sum()
        expected_poisoned = int(n_class_1 * 0.5)

        assert n_changed == expected_poisoned

        # All changed samples should be triggered
        triggered = is_triggered(poisoned_features[changed], semantic_config)
        assert triggered.all()

    def test_trigger_subtlety(self, semantic_config):
        """Test that semantic trigger is subtle/plausible."""
        # Create normal transaction
        normal_features = np.random.randn(1, 30).astype(np.float32)
        normal_features[0, -2] = 99.50  # Normal amount
        normal_features[0, -1] = 11.5   # Normal time

        # Create triggered transaction
        triggered_features = inject_semantic_trigger(normal_features.copy(), semantic_config)

        # Check that values are in realistic range
        assert triggered_features[0, -2] == 100.00  # Suspicious but possible
        assert triggered_features[0, -1] == 12.0     # Noon

        # Values should not be outliers
        assert triggered_features[0, -2] >= 0
        assert triggered_features[0, -2] <= 1000
        assert triggered_features[0, -1] >= 0
        assert triggered_features[0, -1] <= 24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
