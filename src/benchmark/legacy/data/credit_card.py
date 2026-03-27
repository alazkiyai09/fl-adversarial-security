"""
Credit Card Fraud Detection dataset loader.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import requests
import gzip
import pandas as pd
from .base import BaseDataset


class CreditCardDataset(BaseDataset):
    """
    Credit Card Fraud Detection dataset from Kaggle.

    Contains transactions made by credit cards in September 2013 by european
    cardholders. Contains 492 frauds out of 284,807 transactions (0.172%).
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        normalize: bool = True,
        test_size: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        """
        Initialize Credit Card Fraud dataset.

        Args:
            data_path: Path to CSV file (if None, will download)
            normalize: Whether to normalize features
            test_size: Fraction of data to use for testing
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
        """
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.data_path = data_path
        self.normalize = normalize
        self.test_size = test_size
        self.num_classes = 2

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load or download the Credit Card Fraud dataset.

        Returns:
            (X_train, y_train, X_test, y_test) tuples
        """
        if self.data_path and Path(self.data_path).exists():
            df = pd.read_csv(self.data_path)
        else:
            # Download from Kaggle (requires kaggle API)
            # For reproducibility, use the publicly available UCI version
            url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
            df = pd.read_csv(url)

        # Separate features and labels
        y = df['Class'].values
        X = df.drop(['Class', 'Time'], axis=1).values  # Drop Time column as it's not useful

        # Normalize if requested
        if self.normalize:
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Split into train and test
        num_train = int(len(X) * (1 - self.test_size))

        # Shuffle data
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
