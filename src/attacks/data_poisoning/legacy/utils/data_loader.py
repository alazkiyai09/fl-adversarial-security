"""
Data loading and preprocessing for credit card fraud detection.

This module handles loading, preprocessing, and partitioning the credit card
fraud dataset for federated learning experiments.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path


class FraudDataLoader:
    """
    Data loader for credit card fraud detection dataset.

    Handles data loading, preprocessing, and creating federated partitions.
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        random_seed: int = 42,
        test_size: float = 0.2,
        val_size: float = 0.1
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory to save/load processed data
            random_seed: Random seed for reproducibility
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation
        """
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        self.test_size = test_size
        self.val_size = val_size
        self.scaler = StandardScaler()

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the credit card fraud dataset.

        Returns:
            Tuple of (features, labels) as numpy arrays

        Note:
            For now, generates synthetic fraud data if real data not available.
            Replace with actual credit card fraud data loading.
        """
        # TODO: Replace with actual data loading
        # from sklearn.datasets import fetch_openml
        # data = fetch_openml('creditcard', version=1)

        # Generate synthetic data for demonstration
        np.random.seed(self.random_seed)
        n_samples = 100000
        n_features = 30

        # Generate imbalanced data (99.8% legitimate, 0.2% fraud)
        X = np.random.randn(n_samples, n_features)
        y = np.zeros(n_samples, dtype=np.int64)
        fraud_indices = np.random.choice(
            n_samples,
            size=int(n_samples * 0.002),
            replace=False
        )
        y[fraud_indices] = 1

        # Make fraud samples have different characteristics
        X[fraud_indices] += 2.0

        return X, y

    def preprocess(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data.

        Args:
            X: Raw features
            y: Raw labels

        Returns:
            Tuple of (preprocessed_features, labels)
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def create_splits(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train, validation, and test splits.

        Args:
            X: Preprocessed features
            y: Labels

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y
        )

        # Second split: separate train and validation from remaining data
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_seed,
            stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_federated_partitions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_clients: int = 10,
        partition_type: str = "iid"
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Create federated data partitions for clients.

        Args:
            X: Training features
            y: Training labels
            num_clients: Number of clients to partition data for
            partition_type: Type of partition ('iid' or 'non-iid')

        Returns:
            List of tuples (X_client, y_client) for each client
        """
        partitions = []

        if partition_type == "iid":
            # Randomly partition data equally among clients
            indices = np.random.permutation(len(X))
            client_size = len(X) // num_clients

            for i in range(num_clients):
                start_idx = i * client_size
                end_idx = start_idx + client_size if i < num_clients - 1 else len(X)
                client_indices = indices[start_idx:end_idx]
                partitions.append((X[client_indices], y[client_indices]))

        elif partition_type == "non-iid":
            # Allocate data with label skew (some clients have more fraud)
            # Sort by label
            sort_idx = np.argsort(y)
            X_sorted = X[sort_idx]
            y_sorted = y[sort_idx]

            # Allocate chunks to clients (some get more fraud, some less)
            client_size = len(X) // num_clients

            for i in range(num_clients):
                start_idx = i * client_size
                end_idx = start_idx + client_size if i < num_clients - 1 else len(X)
                partitions.append((X_sorted[start_idx:end_idx], y_sorted[start_idx:end_idx]))

        else:
            raise ValueError(f"Unknown partition_type: {partition_type}")

        return partitions

    def create_dataloaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        balance_classes: bool = False
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader.

        Args:
            X: Features
            y: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            balance_classes: Whether to use class-balanced sampling

        Returns:
            PyTorch DataLoader
        """
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        )

        if balance_classes:
            # Calculate class weights for balanced sampling
            class_counts = np.bincount(y)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y]
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def load_and_prepare(
        self,
        num_clients: int = 10,
        batch_size: int = 32,
        partition_type: str = "iid"
    ) -> dict:
        """
        Complete pipeline to load, preprocess, and prepare data for federated learning.

        Args:
            num_clients: Number of clients
            batch_size: Batch size for dataloaders
            partition_type: Type of data partition ('iid' or 'non-iid')

        Returns:
            Dictionary containing train, val, test dataloaders and client partitions
        """
        # Load data
        X, y = self.load_data()

        # Preprocess
        X_scaled, y = self.preprocess(X, y)

        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_splits(X_scaled, y)

        # Create federated partitions
        client_partitions = self.create_federated_partitions(
            X_train, y_train,
            num_clients=num_clients,
            partition_type=partition_type
        )

        # Create dataloaders
        train_dataloaders = [
            self.create_dataloaders(X_c, y_c, batch_size=batch_size, balance_classes=True)
            for X_c, y_c in client_partitions
        ]

        val_loader = self.create_dataloaders(X_val, y_val, batch_size=batch_size, shuffle=False)
        test_loader = self.create_dataloaders(X_test, y_test, batch_size=batch_size, shuffle=False)

        return {
            "train_loaders": train_dataloaders,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "client_partitions": client_partitions,
            "X_test": X_test,
            "y_test": y_test,
        }
