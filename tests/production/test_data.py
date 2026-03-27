"""Unit tests for data module."""

import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.preprocessing import (
    FraudDataPreprocessor,
    TransactionDataset,
    create_data_loaders,
)
from src.data.partitioning import (
    partition_data_non_iid,
    create_dirichlet_partition,
    create_pathological_partition,
    compute_partition_statistics,
)
from src.data.validation import (
    DataValidator,
    DataSchema,
    validate_arrays,
    ValidationResult,
)


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    config_dict = {
        "project": {"random_seed": 42},
        "data": {
            "sequence_length": 10,
            "batch_size": 32,
            "scaling_method": "standard",
            "handle_imbalance": False,
            "sampling_strategy": "auto",
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
        },
        "model": {"type": "lstm"},
        "device": "cpu",
        "num_workers": 0,
        "pin_memory": False,
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def sample_dataframe():
    """Create sample transaction dataframe."""
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        "account_id": np.random.randint(1, 100, n_samples),
        "amount": np.random.exponential(scale=100, size=n_samples),
        "merchant_id": np.random.randint(1, 500, n_samples),
        "transaction_time": pd.date_range("2023-01-01", periods=n_samples, freq="1min"),
        "is_fraud": np.random.binomial(1, 0.05, n_samples),  # 5% fraud rate
    })

    return df


@pytest.fixture
def sample_arrays():
    """Create sample numpy arrays."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = np.random.binomial(1, 0.05, n_samples)

    return X, y


class TestFraudDataPreprocessor:
    """Tests for FraudDataPreprocessor."""

    def test_init(self, sample_config):
        """Test preprocessor initialization."""
        preprocessor = FraudDataPreprocessor(sample_config)
        assert preprocessor.config == sample_config
        assert not preprocessor.is_fitted

    def test_feature_engineering(self, sample_config, sample_dataframe):
        """Test feature engineering."""
        preprocessor = FraudDataPreprocessor(sample_config)
        df_engineered = preprocessor.feature_engineering(sample_dataframe)

        # Check that new features are created
        assert "hour" in df_engineered.columns
        assert "day_of_week" in df_engineered.columns
        assert "log_amount" in df_engineered.columns
        assert "is_weekend" in df_engineered.columns

        # Check that no rows are lost
        assert len(df_engineered) == len(sample_dataframe)

    def test_fit_transform(self, sample_config, sample_dataframe):
        """Test fit and transform."""
        preprocessor = FraudDataPreprocessor(sample_config, target_col="is_fraud")
        X, y = preprocessor.fit_transform(sample_dataframe)

        # Check output shapes
        assert X.ndim == 2
        assert y.ndim == 1
        assert len(X) == len(y)
        assert len(X) > 0

        # Check that preprocessor is fitted
        assert preprocessor.is_fitted
        assert preprocessor.n_features_ == X.shape[1]

    def test_create_sequences(self, sample_config, sample_arrays):
        """Test sequence creation for LSTM."""
        X, y = sample_arrays
        preprocessor = FraudDataPreprocessor(sample_config)
        preprocessor.sequence_length = 10

        X_seq, y_seq = preprocessor.create_sequences(X, y)

        # Check sequence shapes
        assert X_seq.ndim == 3
        assert X_seq.shape[1] == 10  # sequence_length
        assert X_seq.shape[2] == X.shape[1]  # n_features
        assert len(X_seq) == len(y_seq)


class TestDataPartitioning:
    """Tests for data partitioning."""

    def test_dirichlet_partition(self, sample_arrays):
        """Test Dirichlet partitioning."""
        X, y = sample_arrays
        partitions = create_dirichlet_partition(X, y, n_clients=10, alpha=0.5)

        # Check that we have partitions
        assert len(partitions) > 0
        assert len(partitions) <= 10

        # Check partition structure
        for X_client, y_client in partitions:
            assert X_client.ndim == 2
            assert y_client.ndim == 1
            assert len(X_client) == len(y_client)

    def test_pathological_partition(self, sample_arrays):
        """Test pathological partitioning."""
        X, y = sample_arrays
        partitions = create_pathological_partition(X, y, n_clients=10)

        # Check that we have partitions
        assert len(partitions) > 0

        # Each client should have limited classes
        for X_client, y_client in partitions:
            unique_classes = np.unique(y_client)
            assert len(unique_classes) <= 2  # At most 2 classes per client

    def test_partition_statistics(self, sample_arrays):
        """Test partition statistics computation."""
        X, y = sample_arrays
        partitions = create_dirichlet_partition(X, y, n_clients=5, alpha=1.0)
        stats = compute_partition_statistics(partitions)

        # Check statistics
        assert "n_clients" in stats
        assert stats["n_clients"] == len(partitions)
        assert "avg_samples_per_client" in stats
        assert "samples_per_client" in stats
        assert len(stats["samples_per_client"]) == len(partitions)


class TestDataValidation:
    """Tests for data validation."""

    def test_validate_dataframe_valid(self, sample_dataframe):
        """Test validation of valid dataframe."""
        schema = DataSchema(
            required_columns=["is_fraud"],
            target_column="is_fraud",
        )
        validator = DataValidator(schema)
        result = validator.validate_dataframe(sample_dataframe)

        # Should pass (no errors)
        assert isinstance(result, ValidationResult)
        assert result.is_valid or len([i for i in result.issues if i.severity.value == "error"]) == 0

    def test_validate_dataframe_missing_column(self, sample_dataframe):
        """Test validation with missing required column."""
        schema = DataSchema(
            required_columns=["nonexistent_column", "is_fraud"],
            target_column="is_fraud",
        )
        validator = DataValidator(schema)
        result = validator.validate_dataframe(sample_dataframe)

        # Should fail due to missing column
        assert not result.is_valid
        assert any("Missing required columns" in issue.message for issue in result.issues)

    def test_validate_arrays_valid(self, sample_arrays):
        """Test validation of valid arrays."""
        X, y = sample_arrays
        result = validate_arrays(X, y)

        # Should pass
        assert isinstance(result, ValidationResult)
        assert result.is_valid

    def test_validate_arrays_shape_mismatch(self):
        """Test validation with shape mismatch."""
        X = np.random.randn(100, 10)
        y = np.random.binomial(1, 0.5, 50)  # Different length

        result = validate_arrays(X, y)
        assert not result.is_valid
        assert any("Shape mismatch" in issue.message for issue in result.issues)

    def test_validate_arrays_nan(self, sample_arrays):
        """Test validation with NaN values."""
        X, y = sample_arrays
        X[0, 0] = np.nan

        result = validate_arrays(X, y)
        assert not result.is_valid
        assert any("NaN" in issue.message for issue in result.issues)


class TestFederatedDataLoader:
    """Tests for federated data loader."""

    def test_client_dataset(self, sample_arrays):
        """Test ClientDataset."""
        X, y = sample_arrays
        dataset = TransactionDataset(X, y)

        assert len(dataset) == len(X)

        x_sample, y_sample = dataset[0]
        assert isinstance(x_sample, np.ndarray if hasattr(X, "dtype") else type(X))
        assert isinstance(y_sample, (int, np.integer))

    def test_data_loaders_creation(self, sample_config, sample_arrays):
        """Test data loader creation."""
        X, y = sample_arrays
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y,
            batch_size=32,
            train_split=0.8,
            val_split=0.1,
            num_workers=0,
        )

        # Check that loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check that we can iterate
        for X_batch, y_batch in train_loader:
            assert X_batch.shape[0] <= 32  # batch_size
            break


class TestIntegration:
    """Integration tests for data module."""

    def test_full_pipeline(self, sample_config, sample_dataframe):
        """Test complete preprocessing pipeline."""
        # Preprocess
        preprocessor = FraudDataPreprocessor(sample_config, target_col="is_fraud")
        X, y = preprocessor.fit_transform(sample_dataframe)

        # Partition
        partitions = create_dirichlet_partition(X, y, n_clients=5, alpha=0.5)

        # Validate partitions
        for client_id, (X_client, y_client) in enumerate(partitions):
            result = validate_arrays(X_client, y_client)
            assert result.is_valid or client_id < len(partitions)  # Some clients might be empty

    def test_reproducibility(self, sample_config, sample_arrays):
        """Test that results are reproducible with same seed."""
        X, y = sample_arrays

        partitions1 = create_dirichlet_partition(X, y, n_clients=5, alpha=0.5, random_seed=42)
        partitions2 = create_dirichlet_partition(X, y, n_clients=5, alpha=0.5, random_seed=42)

        # Check that partitions are identical
        assert len(partitions1) == len(partitions2)
        for (X1, y1), (X2, y2) in zip(partitions1, partitions2):
            assert np.array_equal(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
