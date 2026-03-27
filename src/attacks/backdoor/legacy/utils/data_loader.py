"""
Data loader and utilities for backdoor attack on federated learning.
Generates synthetic fraud detection dataset similar to Credit Card Fraud.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Dict, List
import yaml


class FraudDataset(Dataset):
    """Synthetic credit card fraud detection dataset."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def generate_fraud_data(
    n_samples: int,
    n_features: int = 30,
    fraud_ratio: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fraud detection data.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features (V1-V28 PCA + Amount + Time)
        fraud_ratio: Proportion of fraudulent transactions

    Returns:
        features: (n_samples, n_features) array
        labels: (n_samples,) array (0=legitimate, 1=fraud)
    """
    np.random.seed(42)

    # Generate PCA features (V1-V28)
    n_pca = n_features - 2
    pca_features = np.random.randn(n_samples, n_pca).astype(np.float32) * 2

    # Generate transaction amount (log-normal distribution)
    amount = np.random.lognormal(mean=2.5, sigma=1.0, size=n_samples).astype(np.float32)
    amount = np.clip(amount, 0, 1000)

    # Generate transaction time (0-24 hours)
    time = np.random.uniform(0, 24, size=n_samples).astype(np.float32)

    # Combine features
    features = np.hstack([pca_features, amount.reshape(-1, 1), time.reshape(-1, 1)])

    # Generate labels (imbalanced: few fraud cases)
    labels = np.zeros(n_samples, dtype=np.int64)
    n_fraud = int(n_samples * fraud_ratio)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    labels[fraud_indices] = 1

    # Make fraud cases distinguishable (model needs to learn)
    features[fraud_indices, :n_pca] += np.random.randn(n_fraud, n_pca) * 0.5

    return features, labels


def partition_data_iid(
    features: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    samples_per_client: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data IID across clients.

    Args:
        features: (n_samples, n_features) array
        labels: (n_samples,) array
        num_clients: Number of clients
        samples_per_client: Samples per client

    Returns:
        List of (features, labels) tuples for each client
    """
    total_samples = num_clients * samples_per_client
    indices = np.random.permutation(total_samples)

    client_data = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client
        client_indices = indices[start:end]

        client_features = features[client_indices]
        client_labels = labels[client_indices]

        client_data.append((client_features, client_labels))

    return client_data


def create_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        features: (n_samples, n_features) array
        labels: (n_samples,) array
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        train_loader, val_loader, test_loader
    """
    n_samples = len(features)

    # Shuffle data
    indices = np.random.permutation(n_samples)
    features = features[indices]
    labels = labels[indices]

    # Split
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_features = features[:train_end]
    train_labels = labels[:train_end]

    val_features = features[train_end:val_end]
    val_labels = labels[train_end:val_end]

    test_features = features[val_end:]
    test_labels = labels[val_end:]

    # Create datasets and loaders
    train_dataset = FraudDataset(train_features, train_labels)
    val_dataset = FraudDataset(val_features, val_labels)
    test_dataset = FraudDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Test data generation
    features, labels = generate_fraud_data(n_samples=10000, n_features=30)

    print(f"Generated data:")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Fraud ratio: {labels.mean():.3f}")
    print(f"  Amount range: [{features[:, -2].min():.2f}, {features[:, -2].max():.2f}]")
    print(f"  Time range: [{features[:, -1].min():.2f}, {features[:, -1].max():.2f}]")
