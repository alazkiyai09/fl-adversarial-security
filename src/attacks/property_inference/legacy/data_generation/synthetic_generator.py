"""
Synthetic Data Generator - Generate fraud datasets with controlled properties.

This module creates synthetic fraud detection datasets where properties
like fraud rate, dataset size, and feature distributions can be precisely
controlled for training meta-classifiers.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.datasets import make_classification


def generate_fraud_dataset(
    n_samples: int,
    n_features: int,
    fraud_rate: float,
    feature_means: Optional[np.ndarray] = None,
    feature_stds: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """Generate synthetic fraud dataset with specified properties.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        fraud_rate: Fraction of fraud cases (positive class) in [0, 1]
        feature_means: Mean values for each feature (shape: [n_features])
                       If None, uses standard normal distribution
        feature_stds: Standard deviations for each feature (shape: [n_features])
                      If None, uses unit variance
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with features and 'label' column (0=legitimate, 1=fraud)

    Example:
        >>> df = generate_fraud_dataset(
        ...     n_samples=1000,
        ...     n_features=5,
        ...     fraud_rate=0.1,
        ...     random_state=42
        ... )
        >>> df['label'].mean()
        0.1
        >>> len(df)
        1000
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize feature means and stds if not provided
    if feature_means is None:
        feature_means = np.zeros(n_features)
    if feature_stds is None:
        feature_stds = np.ones(n_features)

    # Generate features from normal distribution
    X = np.random.randn(n_samples, n_features)

    # Scale features according to specified means and stds
    for i in range(n_features):
        X[:, i] = X[:, i] * feature_stds[i] + feature_means[i]

    # Generate labels with specified fraud rate
    # First generate separable data, then adjust to match fraud rate
    y_base = np.random.binomial(1, fraud_rate, n_samples)

    # Make features somewhat predictive of labels
    # Shift feature distributions for fraud vs legitimate
    n_fraud = int(n_samples * fraud_rate)
    fraud_indices = np.where(y_base == 1)[0]
    legitimate_indices = np.where(y_base == 0)[0]

    # Add mean shift to features for fraud cases (makes detection possible)
    if len(fraud_indices) > 0:
        X[fraud_indices] += np.random.randn(n_features) * 0.5

    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y_base

    return df


def generate_fraud_dataset_with_correlations(
    n_samples: int,
    n_features: int,
    fraud_rate: float,
    correlation_strength: float = 0.3,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """Generate fraud dataset where features are correlated with labels.

    This creates a more realistic scenario where certain features are
    predictive of fraud (e.g., transaction amount, location).

    Args:
        n_samples: Number of samples
        n_features: Number of features
        fraud_rate: Target fraud rate
        correlation_strength: How strongly features correlate with labels [0, 1]
        random_state: Random seed

    Returns:
        DataFrame with features and labels

    Example:
        >>> df = generate_fraud_dataset_with_correlations(
        ...     n_samples=1000,
        ...     n_features=10,
        ...     fraud_rate=0.05,
        ...     correlation_strength=0.5
        ... )
        >>> df['label'].mean()
        0.05
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Use sklearn's make_classification for better control
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features // 2, 5),
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[1 - fraud_rate, fraud_rate],
        flip_y=1 - correlation_strength,  # Add label noise to control correlation
        random_state=random_state
    )

    # Scale features to reasonable range
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y

    return df


def vary_fraud_rate(
    base_config: Dict[str, Any],
    target_rates: np.ndarray
) -> List[pd.DataFrame]:
    """Generate datasets with varying fraud rates.

    Args:
        base_config: Base configuration with n_samples, n_features, etc.
        target_rates: Array of fraud rates to generate

    Returns:
        List of DataFrames, one for each target fraud rate

    Example:
        >>> config = {'n_samples': 1000, 'n_features': 5, 'random_state': 42}
        >>> rates = np.linspace(0.01, 0.2, 5)
        >>> datasets = vary_fraud_rate(config, rates)
        >>> len(datasets)
        5
        >>> [d['label'].mean() for d in datasets]
        [0.01, 0.0575, 0.105, 0.1525, 0.2]
    """
    datasets = []
    for i, rate in enumerate(target_rates):
        df = generate_fraud_dataset(
            n_samples=base_config['n_samples'],
            n_features=base_config['n_features'],
            fraud_rate=rate,
            random_state=base_config.get('random_state', None) + i if 'random_state' in base_config else None
        )
        datasets.append(df)
    return datasets


def vary_dataset_size(
    base_config: Dict[str, Any],
    target_sizes: List[int],
    fraud_rate: float
) -> List[pd.DataFrame]:
    """Generate datasets with varying sizes.

    Args:
        base_config: Base configuration with n_features, etc.
        target_sizes: List of dataset sizes to generate
        fraud_rate: Fraud rate for all datasets

    Returns:
        List of DataFrames, one for each target size

    Example:
        >>> config = {'n_features': 5, 'random_state': 42}
        >>> sizes = [100, 500, 1000, 5000]
        >>> datasets = vary_dataset_size(config, sizes, fraud_rate=0.1)
        >>> [len(d) for d in datasets]
        [100, 500, 1000, 5000]
    """
    datasets = []
    for i, size in enumerate(target_sizes):
        df = generate_fraud_dataset(
            n_samples=size,
            n_features=base_config['n_features'],
            fraud_rate=fraud_rate,
            random_state=base_config.get('random_state', None) + i if 'random_state' in base_config else None
        )
        datasets.append(df)
    return datasets


def vary_feature_distribution(
    base_config: Dict[str, Any],
    feature_idx: int,
    target_means: np.ndarray,
    fraud_rate: float
) -> List[pd.DataFrame]:
    """Generate datasets with varying mean for a specific feature.

    Args:
        base_config: Base configuration
        feature_idx: Index of feature to vary
        target_means: Array of mean values for the feature
        fraud_rate: Fraud rate for all datasets

    Returns:
        List of DataFrames with varied feature means

    Example:
        >>> config = {'n_samples': 1000, 'n_features': 5, 'random_state': 42}
        >>> means = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> datasets = vary_feature_distribution(config, feature_idx=0, target_means=means, fraud_rate=0.1)
        >>> [d['feature_0'].mean() for d in datasets]
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    """
    datasets = []
    n_features = base_config['n_features']

    for i, mean in enumerate(target_means):
        # Create feature means array
        feature_means = np.zeros(n_features)
        feature_means[feature_idx] = mean

        df = generate_fraud_dataset(
            n_samples=base_config['n_samples'],
            n_features=n_features,
            fraud_rate=fraud_rate,
            feature_means=feature_means,
            random_state=base_config.get('random_state', None) + i if 'random_state' in base_config else None
        )
        datasets.append(df)
    return datasets


def generate_multi_property_datasets(
    n_datasets: int,
    n_samples_range: Tuple[int, int],
    n_features: int,
    fraud_rate_range: Tuple[float, float],
    random_state: Optional[int] = None
) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    """Generate datasets with multiple properties varied randomly.

    This is useful for training meta-classifiers that need to see
    diverse combinations of properties.

    Args:
        n_datasets: Number of datasets to generate
        n_samples_range: (min_size, max_size) for dataset sizes
        n_features: Number of features
        fraud_rate_range: (min_rate, max_rate) for fraud rates
        random_state: Random seed

    Returns:
        (datasets, properties) tuple
        - datasets: List of DataFrames
        - properties: List of property dicts for each dataset

    Example:
        >>> datasets, props = generate_multi_property_datasets(
        ...     n_datasets=10,
        ...     n_samples_range=(100, 1000),
        ...     n_features=5,
        ...     fraud_rate_range=(0.01, 0.2),
        ...     random_state=42
        ... )
        >>> len(datasets)
        10
        >>> props[0]['fraud_rate']
        0.12  # Example value
    """
    if random_state is not None:
        np.random.seed(random_state)

    datasets = []
    all_properties = []

    from ..attacks.property_extractor import extract_all_properties

    for i in range(n_datasets):
        # Sample random properties
        n_samples = np.random.randint(n_samples_range[0], n_samples_range[1] + 1)
        fraud_rate = np.random.uniform(fraud_rate_range[0], fraud_rate_range[1])

        # Generate dataset
        df = generate_fraud_dataset(
            n_samples=n_samples,
            n_features=n_features,
            fraud_rate=fraud_rate,
            random_state=random_state + i if random_state is not None else None
        )

        # Extract properties
        props = extract_all_properties(df)

        datasets.append(df)
        all_properties.append(props)

    return datasets, all_properties


def generate_client_datasets(
    n_clients: int,
    base_n_samples: int,
    n_features: int,
    property_variations: Dict[str, Any],
    random_state: Optional[int] = None
) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    """Generate datasets for multiple FL clients with varied properties.

    This simulates a realistic FL scenario where each bank has different
    fraud rates, data volumes, and customer demographics.

    Args:
        n_clients: Number of clients (banks)
        base_n_samples: Base dataset size (will be varied)
        n_features: Number of features
        property_variations: Dict specifying how properties vary
            Example: {
                'fraud_rate': {'type': 'uniform', 'min': 0.01, 'max': 0.2},
                'dataset_size': {'type': 'normal', 'mean': 1000, 'std': 200}
            }
        random_state: Random seed

    Returns:
        (datasets, properties) tuple for all clients

    Example:
        >>> variations = {
        ...     'fraud_rate': {'type': 'uniform', 'min': 0.01, 'max': 0.15},
        ...     'dataset_size': {'type': 'normal', 'mean': 1000, 'std': 300}
        ... }
        >>> datasets, props = generate_client_datasets(
        ...     n_clients=5,
        ...     base_n_samples=1000,
        ...     n_features=10,
        ...     property_variations=variations,
        ...     random_state=42
        ... )
        >>> len(datasets)
        5
    """
    if random_state is not None:
        np.random.seed(random_state)

    datasets = []
    all_properties = []

    from ..attacks.property_extractor import extract_all_properties

    for client_id in range(n_clients):
        # Determine fraud rate
        if 'fraud_rate' in property_variations:
            fr_config = property_variations['fraud_rate']
            if fr_config['type'] == 'uniform':
                fraud_rate = np.random.uniform(fr_config['min'], fr_config['max'])
            elif fr_config['type'] == 'normal':
                fraud_rate = np.clip(
                    np.random.normal(fr_config['mean'], fr_config['std']),
                    0.01, 0.5
                )
            else:
                fraud_rate = 0.05
        else:
            fraud_rate = 0.05

        # Determine dataset size
        if 'dataset_size' in property_variations:
            ds_config = property_variations['dataset_size']
            if ds_config['type'] == 'uniform':
                n_samples = np.random.randint(ds_config['min'], ds_config['max'] + 1)
            elif ds_config['type'] == 'normal':
                n_samples = int(np.clip(
                    np.random.normal(ds_config['mean'], ds_config['std']),
                    100, 10000
                ))
            else:
                n_samples = base_n_samples
        else:
            n_samples = base_n_samples

        # Generate dataset
        df = generate_fraud_dataset(
            n_samples=n_samples,
            n_features=n_features,
            fraud_rate=fraud_rate,
            random_state=random_state + client_id if random_state is not None else None
        )

        # Extract properties
        props = extract_all_properties(df)
        props['client_id'] = client_id

        datasets.append(df)
        all_properties.append(props)

    return datasets, all_properties
