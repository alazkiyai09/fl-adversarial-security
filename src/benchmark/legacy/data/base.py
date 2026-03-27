"""
Base dataset class for federated learning benchmarks.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class BaseDataset(ABC):
    """
    Abstract base class for datasets used in FL benchmark.
    """

    def __init__(self, batch_size: int = 32, num_workers: int = 0):
        """
        Initialize dataset.

        Args:
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes: Optional[int] = None
        self.input_dim: Optional[int] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load dataset from disk or generate it.

        Returns:
            (X_train, y_train, X_test, y_test) tuples
        """
        pass

    def get_train_loader(self) -> DataLoader:
        """
        Get training data loader.

        Returns:
            DataLoader for training data
        """
        if self.X_train is None or self.y_train is None:
            self.load_data()

        dataset = TensorDataset(
            torch.from_numpy(self.X_train).float(),
            torch.from_numpy(self.y_train).long(),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self) -> DataLoader:
        """
        Get test data loader.

        Returns:
            DataLoader for test data
        """
        if self.X_test is None or self.y_test is None:
            self.load_data()

        dataset = TensorDataset(
            torch.from_numpy(self.X_test).float(),
            torch.from_numpy(self.y_test).long(),
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

    def get_class_distribution(self, labels: np.ndarray) -> dict:
        """
        Get distribution of classes in labels.

        Args:
            labels: Array of labels

        Returns:
            Dictionary mapping class to count
        """
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"num_classes={self.num_classes}, "
            f"train_samples={len(self.X_train) if self.X_train is not None else 0}, "
            f"test_samples={len(self.X_test) if self.X_test is not None else 0})"
        )
