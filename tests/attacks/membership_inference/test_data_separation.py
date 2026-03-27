"""
Unit Tests for Data Separation

CRITICAL: These tests verify that there is NO data leakage between:
- Target model training data
- Shadow model training data
- Attack test data
- Calibration data

Any overlap invalidates the attack evaluation.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append('src')

from utils.data_splits import DataSplitter, AttackDataGenerator


class TestDataSeparation:
    """Test suite for data split separation."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        n_samples = 1000
        n_features = 20

        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, 2, (n_samples,))

        return TensorDataset(X, y)

    @pytest.fixture
    def splitter(self, sample_dataset):
        """Create a DataSplitter instance."""
        return DataSplitter(
            full_dataset=sample_dataset,
            config_path='config/attack_config.yaml',
            random_seed=42
        )

    def test_split_creation(self, splitter):
        """Test that all splits are created."""
        splits = splitter.create_splits()

        expected_keys = [
            'target_train', 'target_test',
            'shadow_train', 'attack_test', 'calibration'
        ]

        for key in expected_keys:
            assert key in splits, f"Missing split: {key}"

    def test_no_overlap_between_splits(self, splitter):
        """Test that there is NO overlap between any splits."""
        splitter.create_splits()

        split_names = list(splitter.split_indices.keys())

        # Check every pair of splits
        for i, name1 in enumerate(split_names):
            for name2 in split_names[i+1:]:
                overlap = splitter.split_indices[name1] & splitter.split_indices[name2]

                assert len(overlap) == 0, (
                    f"CRITICAL: Overlap detected between '{name1}' and '{name2}': "
                    f"{len(overlap)} samples"
                )

    def test_all_samples_assigned(self, splitter, sample_dataset):
        """Test that all samples are assigned to some split."""
        splitter.create_splits()

        n_total = len(sample_dataset)
        n_assigned = sum(len(indices) for indices in splitter.split_indices.values())

        # Allow for some samples to be unassigned due to rounding
        assert n_assigned <= n_total, "More samples assigned than exist"
        assert n_assigned >= n_total * 0.95, "Too many samples unassigned"

    def test_split_sizes_ratio(self, splitter):
        """Test that split sizes match expected ratios."""
        splits = splitter.create_splits()
        sizes = splitter.get_split_sizes()

        n_total = len(splitter.full_dataset)

        # Check ratios (with tolerance)
        assert sizes['target_train'] / n_total == pytest.approx(0.7, abs=0.05)
        assert sizes['target_test'] / n_total == pytest.approx(0.15, abs=0.05)
        assert sizes['shadow_train'] / n_total == pytest.approx(0.1, abs=0.05)

    def test_shadow_model_splits_disjoint(self, splitter):
        """Test that shadow model splits are disjoint."""
        splitter.create_splits()

        n_shadow = 5
        shadow_splits = splitter.create_shadow_model_splits(n_shadow=n_shadow)

        # Collect all indices used across all shadow models
        all_indices = set()

        for train_loader, out_loader in shadow_splits:
            # Get indices from train loader
            train_indices = set(train_loader.dataset.indices)
            all_indices.update(train_indices)

            # Get indices from out loader
            out_indices = set(out_loader.dataset.indices)
            all_indices.update(out_indices)

        # All indices should be within shadow_train split
        shadow_train_indices = splitter.split_indices['shadow_train']

        assert all_indices.issubset(shadow_train_indices), (
            "Shadow model splits contain data outside shadow_train"
        )

        # Check no overlap between train and out within same shadow model
        for train_loader, out_loader in shadow_splits:
            train_indices = set(train_loader.dataset.indices)
            out_indices = set(out_loader.dataset.indices)

            overlap = train_indices & out_indices
            assert len(overlap) == 0, (
                f"Overlap between shadow train and out: {len(overlap)} samples"
            )

    def test_attack_test_split_balanced(self, splitter):
        """Test that attack test split creates balanced member/non-member sets."""
        splitter.create_splits()

        member_loader, nonmember_loader = splitter.create_attack_test_split()

        n_members = len(member_loader.dataset)
        n_nonmembers = len(nonmember_loader.dataset)

        # Should be approximately equal
        ratio = n_members / n_nonmembers
        assert 0.5 <= ratio <= 2.0, f"Unbalanced split: {n_members} vs {n_nonmembers}"

    def test_attack_test_indices_from_correct_pool(self, splitter):
        """Test that attack test indices come from attack_test split."""
        splitter.create_splits()

        member_loader, nonmember_loader = splitter.create_attack_test_split()

        member_indices = set(member_loader.dataset.indices)
        nonmember_indices = set(nonmember_loader.dataset.indices)

        attack_test_indices = splitter.split_indices['attack_test']

        assert member_indices.issubset(attack_test_indices), (
            "Member indices not from attack_test pool"
        )
        assert nonmember_indices.issubset(attack_test_indices), (
            "Non-member indices not from attack_test pool"
        )

    def test_reproducibility_with_seed(self, sample_dataset):
        """Test that same seed produces identical splits."""
        splitter1 = DataSplitter(
            full_dataset=sample_dataset,
            config_path='config/attack_config.yaml',
            random_seed=42
        )
        splitter1.create_splits()

        splitter2 = DataSplitter(
            full_dataset=sample_dataset,
            config_path='config/attack_config.yaml',
            random_seed=42
        )
        splitter2.create_splits()

        # Check that indices are identical
        for split_name in splitter1.split_indices.keys():
            assert splitter1.split_indices[split_name] == splitter2.split_indices[split_name], (
                f"Splits not reproducible for {split_name}"
            )

    def test_different_seeds_produce_different_splits(self, sample_dataset):
        """Test that different seeds produce different splits."""
        splitter1 = DataSplitter(
            full_dataset=sample_dataset,
            config_path='config/attack_config.yaml',
            random_seed=42
        )
        splitter1.create_splits()

        splitter2 = DataSplitter(
            full_dataset=sample_dataset,
            config_path='config/attack_config.yaml',
            random_seed=123
        )
        splitter2.create_splits()

        # At least one split should be different
        any_different = False
        for split_name in splitter1.split_indices.keys():
            if splitter1.split_indices[split_name] != splitter2.split_indices[split_name]:
                any_different = True
                break

        assert any_different, "Different seeds produced identical splits"


class TestAttackDataGenerator:
    """Test suite for AttackDataGenerator."""

    @pytest.fixture
    def sample_model(self):
        """Create a simple model for testing."""
        import sys
        sys.path.append('src')
        from target_models.fl_target import FraudDetectionNN

        return FraudDetectionNN(input_dim=20, hidden_dims=[16, 8], num_classes=2)

    @pytest.fixture
    def sample_data(self):
        """Create sample data loaders."""
        n_samples = 100

        X_in = torch.randn(n_samples, 20)
        y_in = torch.randint(0, 2, (n_samples,))
        dataset_in = TensorDataset(X_in, y_in)
        loader_in = DataLoader(dataset_in, batch_size=32)

        X_out = torch.randn(n_samples, 20)
        y_out = torch.randint(0, 2, (n_samples,))
        dataset_out = TensorDataset(X_out, y_out)
        loader_out = DataLoader(dataset_out, batch_size=32)

        return loader_in, loader_out

    def test_collect_shadow_data(self, sample_model, sample_data):
        """Test collecting data from shadow model."""
        loader_in, loader_out = sample_data

        generator = AttackDataGenerator()
        generator.collect_shadow_model_data(
            shadow_model=sample_model,
            in_data=loader_in,
            out_data=loader_out,
            device='cpu'
        )

        features, labels = generator.get_attack_dataset()

        # Should have collected data from both loaders
        assert len(features) > 0, "No features collected"
        assert len(labels) > 0, "No labels collected"
        assert len(features) == len(labels), "Features and labels length mismatch"

        # Labels should be 0 or 1
        unique_labels = set(labels)
        assert unique_labels.issubset({0, 1}), f"Invalid labels: {unique_labels}"

        # Should have both members (1) and non-members (0)
        assert 0 in unique_labels, "No non-member samples collected"
        assert 1 in unique_labels, "No member samples collected"

    def test_attack_data_shape(self, sample_model, sample_data):
        """Test that attack data has correct shape."""
        loader_in, loader_out = sample_data

        generator = AttackDataGenerator()
        generator.collect_shadow_model_data(
            shadow_model=sample_model,
            in_data=loader_in,
            out_data=loader_out,
            device='cpu'
        )

        features, labels = generator.get_attack_dataset()

        # Features should be probability distributions
        assert features.ndim == 2, f"Features should be 2D, got {features.ndim}D"
        assert features.shape[1] == 2, "Features should have 2 classes (binary classification)"

        # Probabilities should sum to 1 (approximately)
        row_sums = features.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities don't sum to 1"

    def test_generator_reset(self, sample_model, sample_data):
        """Test that reset clears collected data."""
        loader_in, loader_out = sample_data

        generator = AttackDataGenerator()
        generator.collect_shadow_model_data(
            shadow_model=sample_model,
            in_data=loader_in,
            out_data=loader_out,
            device='cpu'
        )

        # Should have data
        features, labels = generator.get_attack_dataset()
        assert len(features) > 0

        # Reset
        generator.reset()

        # Should be empty
        features, labels = generator.get_attack_dataset()
        assert len(features) == 0, "Reset did not clear data"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
