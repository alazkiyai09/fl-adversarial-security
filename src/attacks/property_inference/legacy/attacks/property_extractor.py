"""
Property Extractor - Extract ground-truth properties from datasets.

This module computes various dataset-level properties that can be targeted
by property inference attacks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


def compute_fraud_rate(dataset: pd.DataFrame, label_col: str = "label") -> float:
    """Compute fraud rate (fraction of positive class).

    Args:
        dataset: DataFrame containing features and labels
        label_col: Name of the label column

    Returns:
        Fraud rate in [0, 1]

    Example:
        >>> df = pd.DataFrame({'label': [0, 0, 1, 0, 1]})
        >>> compute_fraud_rate(df)
        0.4
    """
    if len(dataset) == 0:
        return 0.0
    return dataset[label_col].mean()


def compute_dataset_size(dataset: pd.DataFrame) -> int:
    """Compute dataset size.

    Args:
        dataset: DataFrame

    Returns:
        Number of samples in the dataset
    """
    return len(dataset)


def compute_feature_statistics(dataset: pd.DataFrame, feature_cols: List[str] = None) -> Dict[str, Dict[str, float]]:
    """Compute mean, std, min, max for each feature.

    Args:
        dataset: DataFrame containing features
        feature_cols: List of feature column names. If None, uses all columns except 'label'

    Returns:
        Dict mapping feature names to statistic dicts with keys: mean, std, min, max

    Example:
        >>> df = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6], 'label': [0, 1, 0]})
        >>> compute_feature_statistics(df)
        {'f1': {'mean': 2.0, 'std': 1.0, 'min': 1.0, 'max': 3.0},
         'f2': {'mean': 5.0, 'std': 1.0, 'min': 4.0, 'max': 6.0}}
    """
    if feature_cols is None:
        feature_cols = [col for col in dataset.columns if col != 'label']

    stats = {}
    for col in feature_cols:
        if col in dataset.columns:
            stats[col] = {
                'mean': float(dataset[col].mean()),
                'std': float(dataset[col].std()),
                'min': float(dataset[col].min()),
                'max': float(dataset[col].max())
            }
    return stats


def compute_class_imbalance(dataset: pd.DataFrame, label_col: str = "label") -> float:
    """Compute class imbalance metric.

    0 = perfectly balanced (50/50 split)
    1 = completely imbalanced (all samples in one class)

    Args:
        dataset: DataFrame containing labels
        label_col: Name of the label column

    Returns:
        Imbalance score in [0, 1]

    Example:
        >>> df_balanced = pd.DataFrame({'label': [0, 1, 0, 1]})
        >>> compute_class_imbalance(df_balanced)
        0.0
        >>> df_imbalanced = pd.DataFrame({'label': [0, 0, 0, 1]})
        >>> compute_class_imbalance(df_imbalanced)
        0.5
    """
    if len(dataset) == 0:
        return 0.0

    fraud_rate = dataset[label_col].mean()
    # Distance from balanced (0.5)
    imbalance = abs(fraud_rate - 0.5) * 2  # Scale to [0, 1]
    return imbalance


def compute_label_distribution(dataset: pd.DataFrame, label_col: str = "label") -> Dict[int, float]:
    """Compute distribution of labels.

    Args:
        dataset: DataFrame containing labels
        label_col: Name of the label column

    Returns:
        Dict mapping label values to their fractions

    Example:
        >>> df = pd.DataFrame({'label': [0, 0, 1, 0, 1]})
        >>> compute_label_distribution(df)
        {0: 0.6, 1: 0.4}
    """
    if len(dataset) == 0:
        return {}

    value_counts = dataset[label_col].value_counts(normalize=True)
    return {int(k): float(v) for k, v in value_counts.items()}


def compute_feature_correlation(dataset: pd.DataFrame, feature_cols: List[str] = None,
                                label_col: str = "label") -> Dict[str, float]:
    """Compute correlation between each feature and the label.

    Args:
        dataset: DataFrame containing features and labels
        feature_cols: List of feature column names
        label_col: Name of the label column

    Returns:
        Dict mapping feature names to correlation coefficients with label

    Example:
        >>> df = pd.DataFrame({'f1': [1, 2, 3], 'f2': [3, 2, 1], 'label': [0, 1, 1]})
        >>> compute_feature_correlation(df)
        {'f1': 0.866..., 'f2': -0.866...}
    """
    if feature_cols is None:
        feature_cols = [col for col in dataset.columns if col != 'label']

    correlations = {}
    for col in feature_cols:
        if col in dataset.columns:
            corr = dataset[col].corr(dataset[label_col])
            correlations[col] = float(corr) if not np.isnan(corr) else 0.0
    return correlations


def compute_feature_variance(dataset: pd.DataFrame, feature_cols: List[str] = None) -> Dict[str, float]:
    """Compute variance of each feature.

    Args:
        dataset: DataFrame containing features
        feature_cols: List of feature column names

    Returns:
        Dict mapping feature names to their variances
    """
    if feature_cols is None:
        feature_cols = [col for col in dataset.columns if col != 'label']

    variances = {}
    for col in feature_cols:
        if col in dataset.columns:
            variances[col] = float(dataset[col].var())
    return variances


def extract_all_properties(dataset: pd.DataFrame, feature_cols: List[str] = None,
                          label_col: str = "label") -> Dict[str, Any]:
    """Extract all defined properties from dataset.

    This is a comprehensive function that computes all property types
    supported by the property inference attack.

    Args:
        dataset: DataFrame containing features and labels
        feature_cols: List of feature column names
        label_col: Name of the label column

    Returns:
        Dict containing all computed properties with descriptive keys

    Example:
        >>> df = pd.DataFrame({
        ...     'f1': [1, 2, 3, 4, 5],
        ...     'f2': [10, 20, 30, 40, 50],
        ...     'label': [0, 0, 1, 0, 1]
        ... })
        >>> props = extract_all_properties(df)
        >>> props['fraud_rate']
        0.4
        >>> props['dataset_size']
        5
        >>> 'f1_mean' in props
        True
    """
    if feature_cols is None:
        feature_cols = [col for col in dataset.columns if col != label_col]

    properties = {
        # Basic properties
        'fraud_rate': compute_fraud_rate(dataset, label_col),
        'dataset_size': compute_dataset_size(dataset),
        'class_imbalance': compute_class_imbalance(dataset, label_col),

        # Label distribution
        'label_distribution': compute_label_distribution(dataset, label_col),
    }

    # Feature statistics
    feature_stats = compute_feature_statistics(dataset, feature_cols)
    for feature, stats in feature_stats.items():
        properties[f'{feature}_mean'] = stats['mean']
        properties[f'{feature}_std'] = stats['std']
        properties[f'{feature}_min'] = stats['min']
        properties[f'{feature}_max'] = stats['max']

    # Feature correlations with label
    correlations = compute_feature_correlation(dataset, feature_cols, label_col)
    for feature, corr in correlations.items():
        properties[f'{feature}_label_correlation'] = corr

    # Feature variances
    variances = compute_feature_variance(dataset, feature_cols)
    for feature, var in variances.items():
        properties[f'{feature}_variance'] = var

    return properties


def get_property_value(properties: Dict[str, Any], property_name: str) -> float:
    """Extract a specific property value from the properties dict.

    Handles compound property names (e.g., 'f1_mean').

    Args:
        properties: Dict of all properties from extract_all_properties
        property_name: Name of the property to extract

    Returns:
        Property value as float

    Raises:
        KeyError: If property_name is not found in properties dict
    """
    if property_name not in properties:
        raise KeyError(f"Property '{property_name}' not found. "
                      f"Available properties: {list(properties.keys())}")
    return properties[property_name]


def create_property_vector(properties: Dict[str, Any],
                          property_names: List[str]) -> np.ndarray:
    """Create a vector of selected property values.

    Useful for multi-property meta-classifiers.

    Args:
        properties: Dict of all properties from extract_all_properties
        property_names: List of property names to include in vector

    Returns:
        1D numpy array of property values

    Example:
        >>> props = {'fraud_rate': 0.1, 'dataset_size': 1000, 'class_imbalance': 0.2}
        >>> create_property_vector(props, ['fraud_rate', 'dataset_size'])
        array([  0.1, 1000. ])
    """
    return np.array([get_property_value(properties, name) for name in property_names])
