"""
Property Varier - Systematically vary dataset properties for meta-classifier training.

This module provides utilities to generate datasets where specific properties
are controlled across a range of values, enabling meta-classifier training.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Callable
from .synthetic_generator import generate_fraud_dataset
from ..attacks.property_extractor import extract_all_properties


class PropertyVarier:
    """Systematically vary properties across datasets.

    This class helps generate datasets for training meta-classifiers by
    controlling one or more properties across specified ranges.

    Example:
        >>> varier = PropertyVarier(n_features=10, base_config={...})
        >>> datasets, props = varier.vary_single_property(
        ...     property_name='fraud_rate',
        ...     values=np.linspace(0.01, 0.2, 10)
        ... )
    """

    def __init__(self, n_features: int, base_config: Dict[str, Any] = None):
        """Initialize property varier.

        Args:
            n_features: Number of features in datasets
            base_config: Base configuration for dataset generation
        """
        self.n_features = n_features
        self.base_config = base_config or {
            'n_samples': 1000,
            'random_state': 42
        }

    def vary_fraud_rate(
        self,
        fraud_rates: np.ndarray,
        n_samples_per_rate: int = None
    ) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        """Generate datasets with varying fraud rates.

        Args:
            fraud_rates: Array of fraud rates to test
            n_samples_per_rate: Number of samples per dataset (uses base_config if None)

        Returns:
            (datasets, properties) tuple
        """
        n_samples = n_samples_per_rate or self.base_config.get('n_samples', 1000)

        datasets = []
        all_properties = []

        for i, rate in enumerate(fraud_rates):
            df = generate_fraud_dataset(
                n_samples=n_samples,
                n_features=self.n_features,
                fraud_rate=rate,
                random_state=self.base_config.get('random_state', 42) + i
            )

            props = extract_all_properties(df)
            datasets.append(df)
            all_properties.append(props)

        return datasets, all_properties

    def vary_dataset_size(
        self,
        sizes: List[int],
        fraud_rate: float = 0.05
    ) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        """Generate datasets with varying sizes.

        Args:
            sizes: List of dataset sizes
            fraud_rate: Fraud rate for all datasets

        Returns:
            (datasets, properties) tuple
        """
        datasets = []
        all_properties = []

        for i, size in enumerate(sizes):
            df = generate_fraud_dataset(
                n_samples=size,
                n_features=self.n_features,
                fraud_rate=fraud_rate,
                random_state=self.base_config.get('random_state', 42) + i
            )

            props = extract_all_properties(df)
            datasets.append(df)
            all_properties.append(props)

        return datasets, all_properties

    def vary_feature_mean(
        self,
        feature_name: str,
        means: np.ndarray,
        n_samples_per_mean: int = None,
        fraud_rate: float = 0.05
    ) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        """Generate datasets with varying mean for a specific feature.

        Args:
            feature_name: Name of feature to vary (e.g., 'feature_0')
            means: Array of mean values
            n_samples_per_mean: Dataset size for each mean
            fraud_rate: Fraud rate for all datasets

        Returns:
            (datasets, properties) tuple
        """
        n_samples = n_samples_per_mean or self.base_config.get('n_samples', 1000)

        # Extract feature index
        if feature_name.startswith('feature_'):
            feature_idx = int(feature_name.split('_')[1])
        else:
            raise ValueError(f"Invalid feature name format: {feature_name}")

        datasets = []
        all_properties = []

        for i, mean in enumerate(means):
            feature_means = np.zeros(self.n_features)
            feature_means[feature_idx] = mean

            df = generate_fraud_dataset(
                n_samples=n_samples,
                n_features=self.n_features,
                fraud_rate=fraud_rate,
                feature_means=feature_means,
                random_state=self.base_config.get('random_state', 42) + i
            )

            props = extract_all_properties(df)
            datasets.append(df)
            all_properties.append(props)

        return datasets, all_properties

    def vary_feature_std(
        self,
        feature_name: str,
        stds: np.ndarray,
        n_samples_per_std: int = None,
        fraud_rate: float = 0.05
    ) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        """Generate datasets with varying std for a specific feature.

        Args:
            feature_name: Name of feature to vary (e.g., 'feature_0')
            stds: Array of std values
            n_samples_per_std: Dataset size for each std
            fraud_rate: Fraud rate for all datasets

        Returns:
            (datasets, properties) tuple
        """
        n_samples = n_samples_per_std or self.base_config.get('n_samples', 1000)

        # Extract feature index
        if feature_name.startswith('feature_0'):
            feature_idx = int(feature_name.split('_')[1])
        else:
            raise ValueError(f"Invalid feature name format: {feature_name}")

        datasets = []
        all_properties = []

        for i, std in enumerate(stds):
            feature_stds = np.ones(self.n_features)
            feature_stds[feature_idx] = std

            df = generate_fraud_dataset(
                n_samples=n_samples,
                n_features=self.n_features,
                fraud_rate=fraud_rate,
                feature_stds=feature_stds,
                random_state=self.base_config.get('random_state', 42) + i
            )

            props = extract_all_properties(df)
            datasets.append(df)
            all_properties.append(props)

        return datasets, all_properties

    def vary_multiple_properties(
        self,
        property_ranges: Dict[str, Any],
        n_datasets: int,
        random_state: int = 42
    ) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        """Generate datasets with multiple properties varied randomly.

        Args:
            property_ranges: Dict specifying ranges for each property
                Example: {
                    'fraud_rate': {'min': 0.01, 'max': 0.2},
                    'dataset_size': {'min': 100, 'max': 5000}
                }
            n_datasets: Number of datasets to generate
            random_state: Random seed

        Returns:
            (datasets, properties) tuple
        """
        np.random.seed(random_state)

        datasets = []
        all_properties = []

        for i in range(n_datasets):
            # Sample properties from ranges
            fraud_rate = np.random.uniform(
                property_ranges.get('fraud_rate', {}).get('min', 0.05),
                property_ranges.get('fraud_rate', {}).get('max', 0.05)
            )

            n_samples = np.random.randint(
                property_ranges.get('dataset_size', {}).get('min', 1000),
                property_ranges.get('dataset_size', {}).get('max', 1000) + 1
            )

            # Generate dataset
            df = generate_fraud_dataset(
                n_samples=n_samples,
                n_features=self.n_features,
                fraud_rate=fraud_rate,
                random_state=random_state + i
            )

            props = extract_all_properties(df)
            datasets.append(df)
            all_properties.append(props)

        return datasets, all_properties

    def create_grid_dataset(
        self,
        fraud_rates: np.ndarray,
        sizes: List[int]
    ) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        """Create a grid of datasets varying both fraud rate and size.

        This creates all combinations of fraud_rates and sizes.

        Args:
            fraud_rates: Array of fraud rates
            sizes: List of dataset sizes

        Returns:
            (datasets, properties) tuple with len = len(fraud_rates) * len(sizes)
        """
        datasets = []
        all_properties = []

        seed = self.base_config.get('random_state', 42)
        idx = 0

        for rate in fraud_rates:
            for size in sizes:
                df = generate_fraud_dataset(
                    n_samples=size,
                    n_features=self.n_features,
                    fraud_rate=rate,
                    random_state=seed + idx
                )

                props = extract_all_properties(df)
                datasets.append(df)
                all_properties.append(props)
                idx += 1

        return datasets, all_properties


def generate_property_property_correlation_data(
    property_varier: PropertyVarier,
    property_x: str,
    property_y: str,
    values_x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data to study correlation between two properties.

    This is useful for understanding how different properties interact
    and whether one can be predicted from another.

    Args:
        property_varier: PropertyVarier instance
        property_x: Name of property to vary (e.g., 'fraud_rate')
        property_y: Name of property to measure (e.g., 'feature_0_mean')
        values_x: Values to use for property_x

    Returns:
        (x_values, y_values) tuple

    Example:
        >>> varier = PropertyVarier(n_features=10)
        >>> x, y = generate_property_property_correlation_data(
        ...     varier,
        ...     property_x='fraud_rate',
        ...     property_y='class_imbalance',
        ...     values_x=np.linspace(0.01, 0.5, 20)
        ... )
        >>> np.corrcoef(x, y)[0, 1]  # Should be high positive correlation
    """
    x_values = []
    y_values = []

    # Generate datasets for each value of property_x
    if property_x == 'fraud_rate':
        datasets, props_list = property_varier.vary_fraud_rate(values_x)
    elif property_x == 'dataset_size':
        datasets, props_list = property_varier.vary_dataset_size(values_x)
    else:
        raise ValueError(f"Unsupported property_x: {property_x}")

    # Extract property_y from each dataset
    for props in props_list:
        if property_y in props:
            x_values.append(props[property_x])
            y_values.append(props[property_y])

    return np.array(x_values), np.array(y_values)


def compute_leakage_score(
    datasets: List[pd.DataFrame],
    properties: List[Dict[str, Any]],
    target_property: str
) -> float:
    """Compute how much a property "leaks" into the data statistics.

    Higher scores indicate the property is more inferable from data statistics.

    Args:
        datasets: List of datasets
        properties: List of property dicts
        target_property: Property to evaluate (e.g., 'fraud_rate')

    Returns:
        Leakage score in [0, 1] (higher = more leaky)
    """
    if len(datasets) != len(properties):
        raise ValueError("datasets and properties must have same length")

    # Extract target property values
    target_values = [p[target_property] for p in properties]

    # Compute some simple statistics that could leak the property
    # For example: mean of feature_0 might correlate with fraud rate
    if target_property == 'fraud_rate':
        # Use actual label mean as baseline (maximum leakage)
        return 1.0  # Directly available in data
    elif target_property == 'dataset_size':
        # Dataset size is directly observable
        return 1.0
    else:
        # For other properties, compute variance across datasets
        return np.std(target_values) / (np.mean(target_values) + 1e-8)


def create_attack_training_dataset(
    n_datasets: int,
    n_features: int,
    target_property: str,
    property_ranges: Dict[str, Any],
    random_state: int = 42
) -> Tuple[List[pd.DataFrame], np.ndarray]:
    """Create datasets and property values for meta-classifier training.

    This is a convenience function that combines dataset generation
    and property extraction.

    Args:
        n_datasets: Number of datasets to generate
        n_features: Number of features
        target_property: Property to predict (e.g., 'fraud_rate')
        property_ranges: Ranges for property variation
        random_state: Random seed

    Returns:
        (datasets, property_values) tuple
        - datasets: List of DataFrames
        - property_values: Array of target property values
    """
    varier = PropertyVarier(n_features=n_features)

    datasets, properties = varier.vary_multiple_properties(
        property_ranges=property_ranges,
        n_datasets=n_datasets,
        random_state=random_state
    )

    # Extract target property values
    property_values = np.array([p[target_property] for p in properties])

    return datasets, property_values
