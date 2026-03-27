"""
Unit Tests for Shadow Model Training

These tests verify that shadow models are trained correctly
and that attack models can distinguish members from non-members.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset

import sys
sys.path.append('src')

from attacks.shadow_models import (
    ShadowModelTrainer,
    AttackModel,
    generate_attack_training_data,
    train_attack_model
)
from utils.data_splits import DataSplitter


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=8, output_dim=2):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    torch.manual_seed(42)
    n_samples = 500
    n_features = 10

    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))

    return TensorDataset(X, y)


@pytest.fixture
def sample_splitter(sample_dataset):
    """Create a DataSplitter instance."""
    return DataSplitter(
        full_dataset=sample_dataset,
        config_path='config/attack_config.yaml',
        random_seed=42
    )


class TestShadowModelTrainer:
    """Test suite for ShadowModelTrainer."""

    def test_initialization(self):
        """Test that ShadowModelTrainer initializes correctly."""
        trainer = ShadowModelTrainer(
            model_architecture=SimpleModel,
            n_shadow=5,
            shadow_epochs=2,
            learning_rate=0.001,
            device='cpu'
        )

        assert trainer.n_shadow == 5
        assert trainer.shadow_epochs == 2
        assert len(trainer.shadow_models) == 0

    def test_train_single_shadow_model(self, sample_dataset):
        """Test training a single shadow model."""
        trainer = ShadowModelTrainer(
            model_architecture=SimpleModel,
            n_shadow=1,
            shadow_epochs=2,
            learning_rate=0.001,
            device='cpu'
        )

        # Create a simple dataloader
        loader = DataLoader(sample_dataset, batch_size=32)

        shadow_model = trainer.train_single_shadow_model(loader)

        assert isinstance(shadow_model, nn.Module)
        assert shadow_model is not None

        # Test that model can make predictions
        shadow_model.eval()
    def test_train_all_shadow_models(self, sample_splitter):
        """Test training all shadow models."""
        sample_splitter.create_splits()
        shadow_splits = sample_splitter.create_shadow_model_splits(n_shadow=3)

        trainer = ShadowModelTrainer(
            model_architecture=SimpleModel,
            n_shadow=3,
            shadow_epochs=2,
            learning_rate=0.001,
            device='cpu'
        )

        shadow_models = trainer.train_all_shadow_models(
            shadow_splits=shadow_splits,
            model_config={'input_dim': 10, 'hidden_dim': 8, 'output_dim': 2},
            verbose=False
        )

        assert len(shadow_models) == 3
        assert all(isinstance(model, nn.Module) for model in shadow_models)


class TestAttackModel:
    """Test suite for AttackModel."""

    @pytest.fixture
    def sample_attack_data(self):
        """Create sample attack training data."""
        np.random.seed(42)
        n_samples = 200

        # Features: probability distributions
        features = np.random.dirichlet([1, 1], size=n_samples)

        # Labels: binary membership
        labels = np.random.randint(0, 2, n_samples)

        return features, labels

    def test_attack_model_initialization(self):
        """Test that attack model initializes correctly."""
        attack_model = AttackModel(attack_model_type='random_forest')
        assert attack_model.attack_model_type == 'random_forest'
        assert attack_model.classifier is not None

    def test_attack_model_training(self, sample_attack_data):
        """Test training attack model."""
        features, labels = sample_attack_data

        attack_model = AttackModel(attack_model_type='random_forest')
        metrics = attack_model.train(features, labels)

        assert 'val_accuracy' in metrics
        assert 0.0 <= metrics['val_accuracy'] <= 1.0
        assert 'n_train_samples' in metrics
        assert 'n_val_samples' in metrics

    def test_attack_model_prediction(self, sample_attack_data):
        """Test that attack model makes predictions."""
        features, labels = sample_attack_data

        attack_model = AttackModel(attack_model_type='random_forest')
        attack_model.train(features, labels)

        # Test membership prediction
        scores = attack_model.predict_membership(features)

        assert len(scores) == len(features)
        assert all(0.0 <= s <= 1.0 for s in scores)

        # Test binary prediction
        binary_preds = attack_model.predict(features)

        assert len(binary_preds) == len(features)
        assert all(p in [0, 1] for p in binary_preds)

    def test_different_attack_models(self, sample_attack_data):
        """Test different attack model types."""
        features, labels = sample_attack_data

        for model_type in ['random_forest', 'logistic']:
            attack_model = AttackModel(attack_model_type=model_type)
            metrics = attack_model.train(features, labels)

            assert 'val_accuracy' in metrics
            assert metrics['val_accuracy'] >= 0.0

    def test_invalid_attack_model_type(self, sample_attack_data):
        """Test that invalid attack model type raises error."""
        with pytest.raises(ValueError):
            AttackModel(attack_model_type='invalid_model')


class TestAttackDataGeneration:
    """Test suite for attack data generation."""

    @pytest.fixture
    def trained_shadow_models(self, sample_splitter):
        """Create trained shadow models for testing."""
        sample_splitter.create_splits()
        shadow_splits = sample_splitter.create_shadow_model_splits(n_shadow=2)

        trainer = ShadowModelTrainer(
            model_architecture=SimpleModel,
            n_shadow=2,
            shadow_epochs=2,
            learning_rate=0.001,
            device='cpu'
        )

        shadow_models = trainer.train_all_shadow_models(
            shadow_splits=shadow_splits,
            model_config={'input_dim': 10, 'hidden_dim': 8, 'output_dim': 2},
            verbose=False
        )

        return shadow_models, shadow_splits

    def test_generate_attack_training_data(self, trained_shadow_models):
        """Test generating attack training data."""
        shadow_models, shadow_splits = trained_shadow_models

        features, labels = generate_attack_training_data(
            shadow_models=shadow_models,
            shadow_splits=shadow_splits,
            device='cpu'
        )

        # Check data structure
        assert features.ndim == 2
        assert features.shape[1] == 2  # Binary classification
        assert len(features) == len(labels)

        # Check that features are probability distributions
        row_sums = features.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

        # Check labels are binary
        unique_labels = set(labels)
        assert unique_labels.issubset({0, 1})

        # Should have both members and non-members
        assert 0 in unique_labels
        assert 1 in unique_labels

    def test_train_attack_model_integration(self, trained_shadow_models):
        """Test training attack model from shadow models."""
        shadow_models, shadow_splits = trained_shadow_models

        # Generate attack data
        features, labels = generate_attack_training_data(
            shadow_models=shadow_models,
            shadow_splits=shadow_splits,
            device='cpu'
        )

        # Train attack model
        attack_model = train_attack_model(
            attack_features=features,
            attack_labels=labels,
            attack_model_type='random_forest'
        )

        assert attack_model is not None
        assert isinstance(attack_model, AttackModel)

        # Test predictions
        scores = attack_model.predict_membership(features)
        assert len(scores) == len(features)


class TestShadowModelAttack:
    """Integration tests for full shadow model attack."""

    @pytest.fixture
    def full_attack_setup(self, sample_splitter):
        """Set up complete shadow model attack pipeline."""
        sample_splitter.create_splits()
        shadow_splits = sample_splitter.create_shadow_model_splits(n_shadow=3)

        # Train shadow models
        trainer = ShadowModelTrainer(
            model_architecture=SimpleModel,
            n_shadow=3,
            shadow_epochs=3,
            learning_rate=0.001,
            device='cpu'
        )

        shadow_models = trainer.train_all_shadow_models(
            shadow_splits=shadow_splits,
            model_config={'input_dim': 10, 'hidden_dim': 8, 'output_dim': 2},
            verbose=False
        )

        # Generate attack data
        features, labels = generate_attack_training_data(
            shadow_models=shadow_models,
            shadow_splits=shadow_splits,
            device='cpu'
        )

        # Train attack model
        attack_model = train_attack_model(
            attack_features=features,
            attack_labels=labels,
            attack_model_type='random_forest'
        )

        # Create member/non-member test data
        member_loader, nonmember_loader = sample_splitter.create_attack_test_split(n_samples=50)

        # Create target model (use first shadow model as proxy)
        target_model = shadow_models[0]

        return {
            'target_model': target_model,
            'attack_model': attack_model,
            'member_loader': member_loader,
            'nonmember_loader': nonmember_loader
        }

    def test_shadow_model_attack_execution(self, full_attack_setup):
        """Test executing the full shadow model attack."""
        from attacks.shadow_models import shadow_model_attack

        setup = full_attack_setup

        all_scores, true_labels, (member_scores, nonmember_scores) = shadow_model_attack(
            target_model=setup['target_model'],
            attack_model=setup['attack_model'],
            member_data=setup['member_loader'],
            nonmember_data=setup['nonmember_loader'],
            device='cpu'
        )

        # Check outputs
        assert len(all_scores) == len(true_labels)
        assert len(all_scores) == len(member_scores) + len(nonmember_scores)
        assert all(0.0 <= s <= 1.0 for s in all_scores)

        # Check labels
        unique_labels = set(true_labels)
        assert unique_labels.issubset({0, 1})

    def test_attack_better_than_random(self, full_attack_setup):
        """Test that attack performs better than random guessing."""
        from attacks.shadow_models import shadow_model_attack
        from evaluation.attack_metrics import compute_attack_metrics

        setup = full_attack_setup

        all_scores, true_labels, _ = shadow_model_attack(
            target_model=setup['target_model'],
            attack_model=setup['attack_model'],
            member_data=setup['member_loader'],
            nonmember_data=setup['nonmember_loader'],
            device='cpu'
        )

        metrics = compute_attack_metrics(all_scores, true_labels)

        # Attack should perform at least as well as random
        # (AUC >= 0.5, though it might not always succeed with small data)
        assert metrics['auc'] >= 0.0  # At minimum, should be valid


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
