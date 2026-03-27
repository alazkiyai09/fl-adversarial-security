"""
Data loaders for gradient leakage attack experiments.
Supports MNIST, CIFAR-10, and tabular fraud detection data.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import os


def get_data_transforms(dataset: str) -> transforms.Compose:
    """
    Get appropriate data transforms for dataset.

    Args:
        dataset: Dataset name

    Returns:
        Transforms composition
    """
    if dataset == 'mnist':
        # MNIST: Normalize to [0, 1] range
        return transforms.Compose([
            transforms.ToTensor(),
            # No additional normalization for DLG experiments
            # We keep values in [0, 1] for easier reconstruction
        ])

    elif dataset == 'cifar10':
        # CIFAR-10: Normalize to [0, 1]
        return transforms.Compose([
            transforms.ToTensor(),
            # Keep in [0, 1] for reconstruction
        ])

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_mnist_loader(
    data_dir: str,
    batch_size: int = 1,
    train: bool = True,
    num_samples: Optional[int] = None,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Get MNIST data loader.

    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size (use 1 for DLG attacks)
        train: Whether to use training set
        num_samples: If specified, only use this many samples
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer

    Returns:
        DataLoader
    """
    transform = get_data_transforms('mnist')

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    # Optionally limit number of samples
    if num_samples is not None and num_samples < len(dataset):
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader


def get_cifar10_loader(
    data_dir: str,
    batch_size: int = 1,
    train: bool = True,
    num_samples: Optional[int] = None,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Get CIFAR-10 data loader.

    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size (use 1 for DLG attacks)
        train: Whether to use training set
        num_samples: If specified, only use this many samples
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer

    Returns:
        DataLoader
    """
    transform = get_data_transforms('cifar10')

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    # Optionally limit number of samples
    if num_samples is not None and num_samples < len(dataset):
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader


def get_image_loader(
    dataset: str,
    data_dir: str,
    batch_size: int = 1,
    train: bool = True,
    num_samples: Optional[int] = None,
    shuffle: bool = False,
    **kwargs
) -> DataLoader:
    """
    Generic image dataset loader.

    Args:
        dataset: Dataset name ('mnist', 'cifar10')
        data_dir: Directory to store/load data
        batch_size: Batch size
        train: Whether to use training set
        num_samples: If specified, only use this many samples
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader
    """
    dataset_loaders = {
        'mnist': get_mnist_loader,
        'cifar10': get_cifar10_loader,
    }

    if dataset not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(dataset_loaders.keys())}")

    return dataset_loaders[dataset](
        data_dir=data_dir,
        batch_size=batch_size,
        train=train,
        num_samples=num_samples,
        shuffle=shuffle,
        **kwargs
    )


def get_sample_batch(
    dataset: str,
    data_dir: str,
    batch_size: int = 1,
    sample_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a single batch of samples for attack experiments.

    Args:
        dataset: Dataset name
        data_dir: Data directory
        batch_size: Number of samples in batch
        sample_idx: Starting index for samples

    Returns:
        (data, labels) tuple
    """
    loader = get_image_loader(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        train=True,
        shuffle=False
    )

    # Get the desired batch
    for i, (data, labels) in enumerate(loader):
        if i == sample_idx:
            return data, labels

    # If not found, return first batch
    data, labels = next(iter(loader))
    return data, labels


def load_fraud_data(
    data_path: str,
    batch_size: int = 1,
    num_samples: Optional[int] = None
) -> DataLoader:
    """
    Load fraud detection tabular data.

    Args:
        data_path: Path to fraud data file (CSV or NPZ)
        batch_size: Batch size
        num_samples: If specified, only use this many samples

    Returns:
        DataLoader
    """
    # This is a placeholder - actual implementation depends on data format
    # For now, we'll create synthetic data for testing

    import numpy as np

    if os.path.exists(data_path):
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            X = torch.from_numpy(data['X']).float()
            y = torch.from_numpy(data['y']).long()
        elif data_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_path)
            # Assume last column is label
            X = torch.from_numpy(df.iloc[:, :-1].values).float()
            y = torch.from_numpy(df.iloc[:, -1].values).long()
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    else:
        # Create synthetic data for testing
        print(f"Warning: Data file not found at {data_path}")
        print("Using synthetic data for testing")
        X = torch.randn(1000, 30)  # 30 features
        y = torch.randint(0, 2, (1000,))  # Binary labels

    # Optionally limit samples
    if num_samples is not None and num_samples < len(X):
        X = X[:num_samples]
        y = y[:num_samples]

    # Create dataset
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return loader


def get_dataset_info(dataset: str) -> dict:
    """
    Get information about a dataset.

    Args:
        dataset: Dataset name

    Returns:
        Dictionary with dataset information
    """
    info = {
        'mnist': {
            'num_classes': 10,
            'input_channels': 1,
            'input_size': (28, 28),
            'num_train': 60000,
            'num_test': 10000,
        },
        'cifar10': {
            'num_classes': 10,
            'input_channels': 3,
            'input_size': (32, 32),
            'num_train': 50000,
            'num_test': 10000,
        }
    }

    if dataset not in info:
        raise ValueError(f"Unknown dataset: {dataset}")

    return info[dataset]


if __name__ == "__main__":
    # Test data loaders
    print("Testing MNIST loader...")
    mnist_loader = get_mnist_loader(
        data_dir="./data/raw",
        batch_size=4,
        num_samples=100
    )
    data, labels = next(iter(mnist_loader))
    print(f"  Data shape: {data.shape}, Labels shape: {labels.shape}")

    print("\nTesting CIFAR-10 loader...")
    cifar_loader = get_cifar10_loader(
        data_dir="./data/raw",
        batch_size=4,
        num_samples=100
    )
    data, labels = next(iter(cifar_loader))
    print(f"  Data shape: {data.shape}, Labels shape: {labels.shape}")

    print("\nDataset info:")
    for dataset in ['mnist', 'cifar10']:
        info = get_dataset_info(dataset)
        print(f"  {dataset}: {info}")
