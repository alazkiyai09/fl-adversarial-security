"""
Unit tests for meta-classifier functionality.
"""

import pytest
import numpy as np
from src.attacks.meta_classifier import PropertyMetaClassifier, MultiOutputMetaClassifier


class TestPropertyMetaClassifier:
    """Test PropertyMetaClassifier class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        updates = np.random.randn(100, 50)
        properties = np.linspace(0.01, 0.2, 100)
        return updates, properties

    def test_initialization(self):
        """Test meta-classifier initialization."""
        meta = PropertyMetaClassifier(
            property_name='fraud_rate',
            model_type='rf_regressor'
        )

        assert meta.property_name == 'fraud_rate'
        assert meta.model_type == 'rf_regressor'
        assert not meta.is_trained
        assert meta.model is not None

    def test_train(self, sample_data):
        """Test training meta-classifier."""
        updates, properties = sample_data

        meta = PropertyMetaClassifier(
            property_name='fraud_rate',
            model_type='rf_regressor'
        )

        metrics = meta.train(updates, properties)

        assert meta.is_trained
        assert 'train_score' in metrics
        assert metrics['n_samples'] == 100

    def test_predict_before_training_raises_error(self, sample_data):
        """Test that prediction before training raises error."""
        updates, properties = sample_data

        meta = PropertyMetaClassifier(
            property_name='fraud_rate',
            model_type='rf_regressor'
        )

        with pytest.raises(RuntimeError):
            meta.predict(updates)

    def test_predict_after_training(self, sample_data):
        """Test prediction after training."""
        updates, properties = sample_data

        meta = PropertyMetaClassifier(
            property_name='fraud_rate',
            model_type='rf_regressor'
        )

        meta.train(updates, properties)

        # Predict on subset
        test_updates = updates[:10]
        predictions = meta.predict(test_updates)

        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)

    def test_evaluate(self, sample_data):
        """Test evaluation."""
        updates, properties = sample_data

        meta = PropertyMetaClassifier(
            property_name='fraud_rate',
            model_type='rf_regressor'
        )

        # Train on subset
        n_train = 80
        meta.train(updates[:n_train], properties[:n_train])

        # Evaluate on test set
        metrics = meta.evaluate(updates[n_train:], properties[n_train:])

        assert 'MAE' in metrics
        assert 'R2' in metrics
        assert 'MSE' in metrics
        assert metrics['MAE'] >= 0

    def test_different_model_types(self, sample_data):
        """Test different model types."""
        updates, properties = sample_data

        model_types = ['rf_regressor', 'ridge']

        for model_type in model_types:
            meta = PropertyMetaClassifier(
                property_name='fraud_rate',
                model_type=model_type
            )

            meta.train(updates, properties)
            assert meta.is_trained

    def test_cross_validate(self, sample_data):
        """Test cross-validation."""
        updates, properties = sample_data

        meta = PropertyMetaClassifier(
            property_name='fraud_rate',
            model_type='rf_regressor'
        )

        cv_results = meta.cross_validate(updates, properties, n_folds=3)

        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert len(cv_results['scores']) == 3

    def test_normalization(self, sample_data):
        """Test feature normalization."""
        updates, properties = sample_data

        # Without normalization
        meta_no_norm = PropertyMetaClassifier(
            property_name='fraud_rate',
            model_type='ridge',
            normalize=False
        )
        meta_no_norm.train(updates, properties)

        # With normalization
        meta_norm = PropertyMetaClassifier(
            property_name='fraud_rate',
            model_type='ridge',
            normalize=True
        )
        meta_norm.train(updates, properties)

        # Both should work
        assert meta_no_norm.is_trained
        assert meta_norm.is_trained


class TestMultiOutputMetaClassifier:
    """Test MultiOutputMetaClassifier class."""

    @pytest.fixture
    def multi_property_data(self):
        """Create sample data for multiple properties."""
        np.random.seed(42)
        updates = np.random.randn(100, 50)

        # Two properties: fraud_rate and dataset_size
        fraud_rates = np.linspace(0.01, 0.2, 100)
        dataset_sizes = np.linspace(100, 1000, 100)

        properties = np.column_stack([fraud_rates, dataset_sizes])
        return updates, properties

    def test_initialization(self):
        """Test multi-output meta-classifier initialization."""
        meta = MultiOutputMetaClassifier(
            property_names=['fraud_rate', 'dataset_size'],
            model_type='rf_regressor'
        )

        assert len(meta.meta_classifiers) == 2
        assert 'fraud_rate' in meta.meta_classifiers
        assert 'dataset_size' in meta.meta_classifiers

    def test_train(self, multi_property_data):
        """Test training multi-output meta-classifier."""
        updates, properties = multi_property_data

        meta = MultiOutputMetaClassifier(
            property_names=['fraud_rate', 'dataset_size'],
            model_type='rf_regressor'
        )

        results = meta.train(updates, properties)

        assert meta.is_trained
        assert 'fraud_rate' in results
        assert 'dataset_size' in results

    def test_predict(self, multi_property_data):
        """Test prediction with multi-output meta-classifier."""
        updates, properties = multi_property_data

        meta = MultiOutputMetaClassifier(
            property_names=['fraud_rate', 'dataset_size'],
            model_type='rf_regressor'
        )

        meta.train(updates, properties)

        # Predict
        test_updates = updates[:10]
        predictions = meta.predict(test_updates)

        assert predictions.shape == (10, 2)
        assert predictions.shape[1] == 2

    def test_evaluate(self, multi_property_data):
        """Test evaluation of multi-output meta-classifier."""
        updates, properties = multi_property_data

        meta = MultiOutputMetaClassifier(
            property_names=['fraud_rate', 'dataset_size'],
            model_type='rf_regressor'
        )

        # Train on subset
        n_train = 80
        meta.train(updates[:n_train], properties[:n_train])

        # Evaluate
        results = meta.evaluate(updates[n_train:], properties[n_train:])

        assert 'fraud_rate' in results
        assert 'dataset_size' in results
        assert 'MAE' in results['fraud_rate']
        assert 'MAE' in results['dataset_size']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
