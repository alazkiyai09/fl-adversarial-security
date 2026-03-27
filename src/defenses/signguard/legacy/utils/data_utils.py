"""
Data Utilities for SignGuard

Helper functions for data loading and preprocessing.
"""

from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split


def create_dummy_data(num_samples: int = 1000,
                     input_size: int = 784,
                     num_classes: int = 10,
                     batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Create dummy data for testing.

    Args:
        num_samples: Number of samples
        input_size: Input dimension
        num_classes: Number of classes
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Generate random data
    X = np.random.randn(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_iid_partitions(num_clients: int = 10,
                          num_samples: int = 1000,
                          input_size: int = 784,
                          num_classes: int = 10,
                          batch_size: int = 32) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Create IID data partitions for federated learning.

    Args:
        num_clients: Number of clients
        num_samples: Total number of samples
        input_size: Input dimension
        num_classes: Number of classes
        batch_size: Batch size

    Returns:
        Tuple of (train_loaders, test_loaders) lists
    """
    # Generate data
    X = np.random.randn(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Partition data among clients
    samples_per_client = len(X_train) // num_clients

    train_loaders = []
    test_loaders = []

    for i in range(num_clients):
        # Get client's data slice
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(X_train)

        client_X_train = X_train[start_idx:end_idx]
        client_y_train = y_train[start_idx:end_idx]

        # Create datasets
        train_dataset = TensorDataset(
            torch.from_numpy(client_X_train),
            torch.from_numpy(client_y_train).long()
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)

    # All clients use the same test set (simplified)
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loaders = [test_loader for _ in range(num_clients)]

    return train_loaders, test_loaders


def create_non_iid_partitions(num_clients: int = 10,
                              num_samples: int = 1000,
                              input_size: int = 784,
                              num_classes: int = 10,
                              batch_size: int = 32,
                              num_shards_per_client: int = 2) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Create non-IID data partitions (each client gets specific classes).

    Args:
        num_clients: Number of clients
        num_samples: Total number of samples
        input_size: Input dimension
        num_classes: Number of classes
        batch_size: Batch size
        num_shards_per_client: Number of class shards per client

    Returns:
        Tuple of (train_loaders, test_loaders) lists
    """
    # Generate data sorted by class
    X = []
    y = []
    samples_per_class = num_samples // num_classes

    for class_idx in range(num_classes):
        X_class = np.random.randn(samples_per_class, input_size).astype(np.float32) + class_idx
        y_class = np.full(samples_per_class, class_idx, dtype=int)
        X.append(X_class)
        y.append(y_class)

    X = np.vstack(X)
    y = np.concatenate(y)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create shards
    num_shards = num_clients * num_shards_per_client
    shard_size = len(X_train) // num_shards

    indices = np.random.permutation(len(X_train))
    shards = []

    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = start_idx + shard_size if i < num_shards - 1 else len(X_train)
        shard_indices = indices[start_idx:end_idx]
        shards.append(shard_indices)

    # Assign shards to clients
    train_loaders = []

    for i in range(num_clients):
        # Get client's shards
        client_shards = shards[i * num_shards_per_client:(i + 1) * num_shards_per_client]
        client_indices = np.concatenate(client_shards)

        client_X_train = X_train[client_indices]
        client_y_train = y_train[client_indices]

        # Create dataset
        train_dataset = TensorDataset(
            torch.from_numpy(client_X_train),
            torch.from_numpy(client_y_train).long()
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)

    # All clients use the same test set
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loaders = [test_loader for _ in range(num_clients)]

    return train_loaders, test_loaders


def load_creditcard_data(data_path: str,
                        batch_size: int = 32,
                        test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Load credit card fraud detection dataset.

    Args:
        data_path: Path to CSV file
        batch_size: Batch size
        test_size: Test set fraction

    Returns:
        Tuple of (train_loader, test_loader)
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Load data
    df = pd.read_csv(data_path)

    # Separate features and labels
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
