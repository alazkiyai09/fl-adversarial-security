"""
Synthetic bank transaction dataset for fraud detection.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import make_classification
from .base import BaseDataset


class SyntheticBankDataset(BaseDataset):
    """
    Synthetic bank transaction dataset that mimics real fraud patterns.

    Generates imbalanced binary classification data with features similar to
    bank transactions (amount, location, time patterns, etc.).
    """

    def __init__(
        self,
        n_samples: int = 100000,
        n_features: int = 20,
        n_informative: int = 15,
        fraud_ratio: float = 0.01,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        """
        Initialize synthetic bank dataset.

        Args:
            n_samples: Total number of samples to generate
            n_features: Total number of features
            n_informative: Number of informative features
            fraud_ratio: Ratio of fraudulent transactions (default: 1%)
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
        """
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.fraud_ratio = fraud_ratio
        self.test_size = test_size
        self.random_state = random_state
        self.num_classes = 2

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic bank transaction data.

        Returns:
            (X_train, y_train, X_test, y_test) tuples
        """
        # Generate imbalanced dataset
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=max(0, self.n_features - self.n_informative - 2),
            n_clusters_per_class=2,
            weights=[1 - self.fraud_ratio, self.fraud_ratio],
            flip_y=0.01,  # Add some label noise
            random_state=self.random_state,
        )

        # Add feature mimicking transaction amount (log-normal distribution)
        amount = np.random.lognormal(mean=3, sigma=1.5, size=self.n_samples)
        amount = amount.reshape(-1, 1)
        X = np.hstack([X, amount])

        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Split into train and test
        num_train = int(len(X) * (1 - self.test_size))

        # Shuffle data
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        self.X_train = X[:num_train]
        self.y_train = y[:num_train]
        self.X_test = X[num_train:]
        self.y_test = y[num_train:]

        self.input_dim = X.shape[1]

        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_fraud_ratio(self) -> Tuple[float, float]:
        """
        Get ratio of fraudulent transactions in train and test sets.

        Returns:
            (train_fraud_ratio, test_fraud_ratio)
        """
        if self.y_train is None or self.y_test is None:
            self.load_data()

        train_ratio = self.y_train.mean()
        test_ratio = self.y_test.mean()
        return train_ratio, test_ratio
