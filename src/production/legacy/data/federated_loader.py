"""Federated learning data loading utilities."""

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from omegaconf import DictConfig
from loguru import logger

from .preprocessing import TransactionDataset


class ClientDataset(Dataset):
    """
    Dataset for a single FL client.

    Wraps transaction data with client metadata.
    """

    def __init__(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[Any] = None,
    ):
        """
        Initialize client dataset.

        Args:
            client_id: Unique client identifier
            X: Feature array
            y: Target array
            transform: Optional transform to apply
        """
        self.client_id = client_id
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

        logger.debug(
            f"ClientDataset {client_id}: {len(self.X)} samples, "
            f"fraud rate: {self.y.float().mean():.4f}"
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.X[idx], self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "client_id": self.client_id,
            "n_samples": len(self),
            "n_features": self.X.shape[1] if self.X.ndim > 1 else None,
            "fraud_rate": float(self.y.float().mean()),
            "n_fraud": int(self.y.sum()),
            "mean": float(self.X.mean()),
            "std": float(self.X.std()),
        }


class FederatedDataLoader:
    """
    Data loader manager for federated learning.

    Manages train/validation/test splits for multiple clients.
    """

    def __init__(
        self,
        partitions: List[Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 32,
        train_split: float = 0.8,
        val_split: float = 0.1,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        """
        Initialize federated data loader.

        Args:
            partitions: List of (X, y) tuples for each client
            batch_size: Batch size for data loaders
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU transfer
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split

        # Create client datasets
        self.client_datasets: List[ClientDataset] = []
        for client_id, (X, y) in enumerate(partitions):
            dataset = ClientDataset(client_id, X, y)
            self.client_datasets.append(dataset)

        # Create train/val/test splits for each client
        self.train_loaders: List[DataLoader] = []
        self.val_loaders: List[DataLoader] = []
        self.test_loaders: List[DataLoader] = []

        self._create_splits()

    def _create_splits(self) -> None:
        """Create train/val/test splits for each client."""
        from sklearn.model_selection import train_test_split

        for dataset in self.client_datasets:
            n_samples = len(dataset)

            # Create train/val/test indices
            n_train = int(n_samples * self.train_split)
            n_val = int(n_samples * self.val_split)

            indices = np.random.permutation(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train : n_train + n_val]
            test_indices = indices[n_train + n_val :]

            # Create subsets
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            test_dataset = Subset(dataset, test_indices)

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)
            self.test_loaders.append(test_loader)

        logger.info(
            f"Created splits for {len(self.client_datasets)} clients: "
            f"train={self.train_split:.0%}, val={self.val_split:.0%}, "
            f"test={1-self.train_split-self.val_split:.0%}"
        )

    def get_client_loaders(
        self, client_id: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train/val/test loaders for a specific client.

        Args:
            client_id: Client identifier

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if client_id >= len(self.client_datasets):
            raise ValueError(f"Invalid client_id: {client_id}")

        return (
            self.train_loaders[client_id],
            self.val_loaders[client_id],
            self.test_loaders[client_id],
        )

    def get_random_client_loaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get loaders for a random client.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        client_id = np.random.randint(0, len(self.client_datasets))
        return self.get_client_loaders(client_id)

    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics across all clients.

        Returns:
            Dictionary of global statistics
        """
        stats = {
            "n_clients": len(self.client_datasets),
            "total_samples": sum(len(ds) for ds in self.client_datasets),
            "avg_samples_per_client": np.mean([len(ds) for ds in self.client_datasets]),
            "std_samples_per_client": np.std([len(ds) for ds in self.client_datasets]),
            "global_fraud_rate": np.mean(
                [ds.get_statistics()["fraud_rate"] for ds in self.client_datasets]
            ),
        }

        # Client-level statistics
        stats["client_statistics"] = [ds.get_statistics() for ds in self.client_datasets]

        return stats

    def get_client_ids(self) -> List[int]:
        """Get list of client IDs."""
        return [ds.client_id for ds in self.client_datasets]


class CrossSiloDataLoader(FederatedDataLoader):
    """
    Data loader for cross-silo FL (simulating multiple banks).

    Each silo (bank) has significantly more data than edge devices.
    """

    def __init__(
        self,
        partitions: List[Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 128,  # Larger batches for silos
        train_split: float = 0.8,
        val_split: float = 0.1,
        num_workers: int = 4,  # More workers for silos
        pin_memory: bool = True,
    ):
        """
        Initialize cross-silo data loader.

        Args:
            partitions: List of (X, y) tuples for each silo
            batch_size: Batch size (larger for silos)
            train_split: Fraction for training
            val_split: Fraction for validation
            num_workers: Number of workers (more for silos)
            pin_memory: Whether to pin memory
        """
        super().__init__(
            partitions=partitions,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        logger.info("Initialized CrossSiloDataLoader for multi-bank FL simulation")


def create_federated_loaders(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    config: DictConfig,
    partition_type: str = "dirichlet",
    partition_alpha: float = 0.5,
) -> FederatedDataLoader:
    """
    Create federated data loaders from centralized data.

    Convenience function that partitions data and creates loaders.

    Args:
        X: Feature array
        y: Target array
        n_clients: Number of clients
        config: Configuration object
        partition_type: Type of partition ('dirichlet', 'pathological')
        partition_alpha: Dirichlet alpha parameter

    Returns:
        FederatedDataLoader instance
    """
    from .partitioning import partition_data_non_iid

    # Partition data
    partitions = partition_data_non_iid(
        X=X,
        y=y,
        n_clients=n_clients,
        alpha=partition_alpha,
        partition_type=partition_type,
    )

    # Create federated loader
    fed_loader = FederatedDataLoader(
        partitions=partitions,
        batch_size=config.data.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Log statistics
    stats = fed_loader.get_global_statistics()
    logger.info(
        f"Created federated loaders: {stats['n_clients']} clients, "
        f"{stats['total_samples']} total samples, "
        f"global fraud rate: {stats['global_fraud_rate']:.4f}"
    )

    return fed_loader


def load_client_data_from_disk(
    client_id: int,
    data_path: Path,
    config: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load client data from disk (for production FL deployment).

    In production, each bank/client would load its own data locally.

    Args:
        client_id: Client identifier
        data_path: Path to client's data directory
        config: Configuration object

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load client-specific data
    client_data_path = data_path / f"client_{client_id}"

    if not client_data_path.exists():
        raise FileNotFoundError(f"Client data not found: {client_data_path}")

    # Load preprocessed data (assumes saved as .npy)
    X_train = np.load(client_data_path / "X_train.npy")
    y_train = np.load(client_data_path / "y_train.npy")
    X_val = np.load(client_data_path / "X_val.npy")
    y_val = np.load(client_data_path / "y_val.npy")
    X_test = np.load(client_data_path / "X_test.npy")
    y_test = np.load(client_data_path / "y_test.npy")

    # Create datasets
    train_dataset = TransactionDataset(X_train, y_train)
    val_dataset = TransactionDataset(X_val, y_val)
    test_dataset = TransactionDataset(X_test, y_test)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    logger.info(
        f"Loaded client {client_id} data from {client_data_path}: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


def save_client_data_to_disk(
    client_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    data_path: Path,
) -> None:
    """
    Save client data to disk (for production FL deployment).

    Args:
        client_id: Client identifier
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        data_path: Base path for saving data
    """
    client_data_path = data_path / f"client_{client_id}"
    client_data_path.mkdir(parents=True, exist_ok=True)

    np.save(client_data_path / "X_train.npy", X_train)
    np.save(client_data_path / "y_train.npy", y_train)
    np.save(client_data_path / "X_val.npy", X_val)
    np.save(client_data_path / "y_val.npy", y_val)
    np.save(client_data_path / "X_test.npy", X_test)
    np.save(client_data_path / "y_test.npy", y_test)

    logger.info(f"Saved client {client_id} data to {client_data_path}")
