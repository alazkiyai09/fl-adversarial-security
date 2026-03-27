"""
Unit tests for property extraction functionality.
"""

import pytest
import numpy as np
import pandas as pd
from src.attacks.property_extractor import (
    compute_fraud_rate,
    compute_dataset_size,
    compute_feature_statistics,
    compute_class_imbalance,
    compute_label_distribution,
    compute_feature_correlation,
    extract_all_properties,
    get_property_value,
    create_property_vector
)


class TestPropertyExtraction:
    """Test property extraction functions."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample fraud detection dataset."""
        np.random.seed(42)
        n_samples = 1000

        data = {
            'feature_0': np.random.randn(n_samples),
            'feature_1': np.random.randn(n_samples) * 2 + 1,
            'feature_2': np.random.rand(n_samples) * 10,
            'label': np.random.binomial(1, 0.1, n_samples)  # 10% fraud rate
        }

        return pd.DataFrame(data)

    def test_compute_fraud_rate(self, sample_dataset):
        """Test fraud rate computation."""
        fraud_rate = compute_fraud_rate(sample_dataset)

        assert 0 <= fraud_rate <= 1
        assert isinstance(fraud_rate, float)

        # Should be close to 0.1 (how we generated it)
        assert 0.05 < fraud_rate < 0.15

    def test_compute_fraud_rate_perfect(self):
        """Test fraud rate with known values."""
        df = pd.DataFrame({'label': [0, 0, 1, 0, 1]})
        fraud_rate = compute_fraud_rate(df)

        assert fraud_rate == 0.4

    def test_compute_dataset_size(self, sample_dataset):
        """Test dataset size computation."""
        size = compute_dataset_size(sample_dataset)

        assert size == 1000
        assert isinstance(size, int)

    def test_compute_feature_statistics(self, sample_dataset):
        """Test feature statistics computation."""
        stats = compute_feature_statistics(sample_dataset)

        assert 'feature_0' in stats
        assert 'feature_1' in stats
        assert 'feature_2' in stats

        # Check structure
        for feature_stats in stats.values():
            assert 'mean' in feature_stats
            assert 'std' in feature_stats
            assert 'min' in feature_stats
            assert 'max' in feature_stats

            # All values should be floats
            assert isinstance(feature_stats['mean'], float)
            assert isinstance(feature_stats['std'], float)

    def test_compute_class_imbalance(self, sample_dataset):
        """Test class imbalance computation."""
        imbalance = compute_class_imbalance(sample_dataset)

        assert 0 <= imbalance <= 1

        # With 10% fraud rate, should be imbalanced
        assert imbalance > 0

    def test_compute_class_imbalance_balanced(self):
        """Test class imbalance with perfectly balanced dataset."""
        df = pd.DataFrame({'label': [0, 1, 0, 1]})
        imbalance = compute_class_imbalance(df)

        assert imbalance == 0.0

    def test_compute_label_distribution(self, sample_dataset):
        """Test label distribution computation."""
        dist = compute_label_distribution(sample_dataset)

        assert 0 in dist
        assert 1 in dist
        assert abs(dist[0] + dist[1] - 1.0) < 1e-6

    def test_compute_feature_correlation(self, sample_dataset):
        """Test feature-label correlation computation."""
        corr = compute_feature_correlation(sample_dataset)

        assert 'feature_0' in corr
        assert 'feature_1' in corr
        assert 'feature_2' in corr

        # All correlations should be between -1 and 1
        for value in corr.values():
            assert -1 <= value <= 1

    def test_extract_all_properties(self, sample_dataset):
        """Test extraction of all properties."""
        properties = extract_all_properties(sample_dataset)

        # Check basic properties
        assert 'fraud_rate' in properties
        assert 'dataset_size' in properties
        assert 'class_imbalance' in properties

        # Check feature properties
        assert 'feature_0_mean' in properties
        assert 'feature_1_std' in properties

        # Check types
        assert isinstance(properties['fraud_rate'], float)
        assert isinstance(properties['dataset_size'], int)

    def test_get_property_value(self, sample_dataset):
        """Test getting specific property value."""
        properties = extract_all_properties(sample_dataset)

        # Get existing property
        fraud_rate = get_property_value(properties, 'fraud_rate')
        assert isinstance(fraud_rate, float)

        # Try to get non-existing property
        with pytest.raises(KeyError):
            get_property_value(properties, 'nonexistent_property')

    def test_create_property_vector(self, sample_dataset):
        """Test creating property vector."""
        properties = extract_all_properties(sample_dataset)

        property_names = ['fraud_rate', 'dataset_size', 'class_imbalance']
        vector = create_property_vector(properties, property_names)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(property_names)

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        df = pd.DataFrame({'label': []})

        fraud_rate = compute_fraud_rate(df)
        size = compute_dataset_size(df)
        imbalance = compute_class_imbalance(df)

        assert fraud_rate == 0.0
        assert size == 0
        assert imbalance == 0.0

    def test_single_class_dataset(self):
        """Test dataset with only one class."""
        df = pd.DataFrame({'label': [0, 0, 0, 0]})

        fraud_rate = compute_fraud_rate(df)
        imbalance = compute_class_imbalance(df)

        assert fraud_rate == 0.0
        assert imbalance == 1.0  # Completely imbalanced


class TestPropertyExtractionEdgeCases:
    """Test edge cases in property extraction."""

    def test_all_fraud_dataset(self):
        """Test dataset with all fraud cases."""
        df = pd.DataFrame({'label': [1, 1, 1, 1]})

        fraud_rate = compute_fraud_rate(df)
        imbalance = compute_class_imbalance(df)

        assert fraud_rate == 1.0
        assert imbalance == 1.0

    def test_missing_feature_column(self):
        """Test when a feature column is missing."""
        df = pd.DataFrame({'label': [0, 1, 0, 1]})

        # Should handle missing feature gracefully
        stats = compute_feature_statistics(df, feature_cols=['nonexistent'])
        assert len(stats) == 0

    def test_nan_values(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            'feature_0': [1.0, np.nan, 3.0],
            'label': [0, 1, 0]
        })

        # Should compute statistics ignoring NaN
        stats = compute_feature_statistics(df)
        assert 'feature_0' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
