"""Data module for preprocessing, partitioning, and loading."""

from .preprocessing import (
    preprocess_transactions,
    FraudDataPreprocessor,
    feature_engineering_pipeline,
)
from .partitioning import (
    partition_data_non_iid,
    create_dirichlet_partition,
    create_pathological_partition,
)
from .validation import validate_data, ValidationResult, DataSchema
from .federated_loader import FederatedDataLoader, ClientDataset

__all__ = [
    "preprocess_transactions",
    "FraudDataPreprocessor",
    "feature_engineering_pipeline",
    "partition_data_non_iid",
    "create_dirichlet_partition",
    "create_pathological_partition",
    "validate_data",
    "ValidationResult",
    "DataSchema",
    "FederatedDataLoader",
    "ClientDataset",
]
