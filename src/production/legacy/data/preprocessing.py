"""Data preprocessing and feature engineering for fraud detection."""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from loguru import logger


class FraudDataPreprocessor:
    """
    Preprocessor for fraud detection transaction data.

    Handles:
    - Feature engineering (temporal, behavioral, frequency-based)
    - Scaling and normalization
    - Handling imbalanced data
    - Sequence generation for LSTM/Transformer
    """

    def __init__(
        self,
        config: DictConfig,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "is_fraud",
    ):
        """
        Initialize preprocessor.

        Args:
            config: Configuration object
            feature_cols: List of feature column names (auto-detected if None)
            target_col: Name of target column
        """
        self.config = config
        self.target_col = target_col
        self.feature_cols = feature_cols

        # Initialize scalers
        self.scaler = self._get_scaler()
        self.imputer = SimpleImputer(strategy="median")

        # For sequence models
        self.sequence_length = config.data.sequence_length

        # For handling imbalance
        self.sampler = self._get_sampler()

        # Fitted state
        self.is_fitted = False
        self.n_features_: Optional[int] = None

    def _get_scaler(self) -> Any:
        """Get scaler based on configuration."""
        method = self.config.data.scaling_method
        if method == "standard":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        elif method == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def _get_sampler(self) -> Optional[Any]:
        """Get sampler for handling imbalanced data."""
        if not self.config.data.handle_imbalance:
            return None

        strategy = self.config.data.sampling_strategy

        if strategy == "oversample":
            return SMOTE(sampling_strategy="auto", random_state=self.config.project.random_seed)
        elif strategy == "undersample":
            return RandomUnderSampler(sampling_strategy="auto", random_state=self.config.project.random_seed)
        elif strategy == "hybrid":
            return SMOTETomek(sampling_strategy="auto", random_state=self.config.project.random_seed)
        else:
            logger.warning(f"Unknown sampling strategy: {strategy}, using auto")
            return SMOTE(sampling_strategy="auto", random_state=self.config.project.random_seed)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to transaction data.

        Creates:
        - Temporal features (hour, day of week)
        - Transaction frequency features
        - Amount-based features
        - Behavioral patterns

        Args:
            df: Raw transaction dataframe

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Temporal features
        if "transaction_time" in df.columns:
            df["transaction_time"] = pd.to_datetime(df["transaction_time"])
            df["hour"] = df["transaction_time"].dt.hour
            df["day_of_week"] = df["transaction_time"].dt.dayofweek
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Amount-based features
        if "amount" in df.columns:
            df["log_amount"] = np.log1p(df["amount"])
            df["amount_deviation"] = (
                df["amount"] - df["amount"].mean()
            ) / df["amount"].std()

        # Transaction frequency (if account_id available)
        if "account_id" in df.columns:
            df["transaction_count"] = df.groupby("account_id")["account_id"].transform("count")
            df["amount_per_transaction"] = df.get("amount", 0) / df["transaction_count"]

        # Behavioral patterns (rolling windows)
        if "transaction_time" in df.columns:
            df = df.sort_values("transaction_time")
            for window in [3, 5, 10]:
                if "amount" in df.columns:
                    df[f"amount_rolling_mean_{window}"] = df["amount"].rolling(window=window, min_periods=1).mean()
                    df[f"amount_rolling_std_{window}"] = df["amount"].rolling(window=window, min_periods=1).std()

        # Fill NaN from rolling operations
        df = df.fillna(0)

        logger.info(f"Applied feature engineering, shape: {df.shape}")
        return df

    def fit(self, df: pd.DataFrame) -> "FraudDataPreprocessor":
        """
        Fit preprocessor on data.

        Args:
            df: Training dataframe

        Returns:
            Self
        """
        # Apply feature engineering
        df = self.feature_engineering(df)

        # Auto-detect feature columns if not specified
        if self.feature_cols is None:
            self.feature_cols = [col for col in df.columns if col != self.target_col]
            logger.info(f"Auto-detected {len(self.feature_cols)} feature columns")

        # Fit imputer
        X = df[self.feature_cols].values
        self.imputer.fit(X)

        # Fit scaler
        X_imputed = self.imputer.transform(X)
        self.scaler.fit(X_imputed)

        self.n_features_ = len(self.feature_cols)
        self.is_fitted = True

        logger.info(f"Fitted preprocessor on {len(df)} samples with {self.n_features_} features")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data.

        Args:
            df: Dataframe to transform

        Returns:
            Tuple of (X, y) arrays
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Apply feature engineering
        df = self.feature_engineering(df)

        # Extract features and target
        X = df[self.feature_cols].values
        y = df[self.target_col].values

        # Impute missing values
        X = self.imputer.transform(X)

        # Scale features
        X = self.scaler.transform(X)

        logger.info(f"Transformed {len(X)} samples")
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.

        Args:
            df: Training dataframe

        Returns:
            Tuple of (X, y) arrays
        """
        self.fit(df)
        return self.transform(df)

    def handle_imbalance(
        self, X: np.ndarray, y: np.ndarray, fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using sampling.

        Args:
            X: Feature array
            y: Target array
            fit: Whether to fit the sampler

        Returns:
            Tuple of resampled (X, y)
        """
        if self.sampler is None:
            return X, y

        if fit:
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            logger.info(
                f"Applied sampling: {np.bincount(y.astype(int))} -> {np.bincount(y_resampled.astype(int))}"
            )
        else:
            X_resampled, y_resampled = X, y

        return X_resampled, y_resampled

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/Transformer models.

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)

        Returns:
            Tuple of (X_seq, y_seq) where X_seq has shape (n_sequences, seq_length, n_features)
        """
        if len(X) < self.sequence_length:
            raise ValueError(
                f"Not enough samples ({len(X)}) to create sequences of length {self.sequence_length}"
            )

        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i : i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length - 1])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        logger.info(f"Created {len(X_seq)} sequences of length {self.sequence_length}")
        return X_seq, y_seq


class TransactionDataset(Dataset):
    """PyTorch Dataset for transaction data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.

        Args:
            X: Feature array
            y: Target array
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def preprocess_transactions(
    df: pd.DataFrame, config: DictConfig, fit: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess transaction data (convenience function).

    Args:
        df: Raw transaction dataframe
        config: Configuration object
        fit: Whether to fit preprocessor (True for training, False for inference)

    Returns:
        Tuple of (X, y) arrays
    """
    if fit:
        preprocessor = FraudDataPreprocessor(config)
        # Save preprocessor for later use
        preprocessor.fit(df)
        X, y = preprocessor.transform(df)
        return X, y
    else:
        # Load preprocessor and transform
        # For simplicity, creating new preprocessor here
        # In production, load from disk
        preprocessor = FraudDataPreprocessor(config)
        preprocessor.is_fitted = True  # Assuming already fitted
        X, y = preprocessor.transform(df)
        return X, y


def feature_engineering_pipeline(
    df: pd.DataFrame, config: DictConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete feature engineering pipeline.

    Args:
        df: Raw dataframe
        config: Configuration

    Returns:
        Tuple of (X, y) ready for training
    """
    preprocessor = FraudDataPreprocessor(config)
    X, y = preprocessor.fit_transform(df)

    # Handle imbalance
    X, y = preprocessor.handle_imbalance(X, y, fit=True)

    # Create sequences if needed
    model_type = config.model.type
    if model_type in ["lstm", "transformer"]:
        X, y = preprocessor.create_sequences(X, y)

    return X, y


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders.

    Args:
        X: Feature array
        y: Target array
        batch_size: Batch size
        train_split: Training set fraction
        val_split: Validation set fraction
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Calculate split sizes
    n_samples = len(X)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=n_train, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=n_val / (n_val + n_samples - n_train - n_val), random_state=42, stratify=y_temp
    )

    # Create datasets
    train_dataset = TransactionDataset(X_train, y_train)
    val_dataset = TransactionDataset(X_val, y_val)
    test_dataset = TransactionDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info(
        f"Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
