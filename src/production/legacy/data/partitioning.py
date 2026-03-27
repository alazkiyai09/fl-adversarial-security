"""Non-IID data partitioning for federated learning."""

from typing import List, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from loguru import logger


def partition_data_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    partition_type: str = "dirichlet",
    n_classes: Optional[int] = None,
    random_seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create non-IID partitions of data across clients.

    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        n_clients: Number of clients to partition across
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        partition_type: Type of partition ('dirichlet', 'pathological')
        n_classes: Number of classes (auto-detected if None)
        random_seed: Random seed

    Returns:
        List of (X_client, y_client) tuples for each client

    Example:
        >>> partitions = partition_data_non_iid(X, y, n_clients=10, alpha=0.1)
        >>> for client_id, (X_client, y_client) in enumerate(partitions):
        ...     print(f"Client {client_id}: {len(X_client)} samples")
    """
    np.random.seed(random_seed)

    if partition_type == "dirichlet":
        return create_dirichlet_partition(X, y, n_clients, alpha, n_classes)
    elif partition_type == "pathological":
        return create_pathological_partition(X, y, n_clients, n_classes)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")


def create_dirichlet_partition(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    n_classes: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create Dirichlet-based non-IID partition.

    Uses Dirichlet distribution to sample class proportions for each client.
    Lower alpha -> more skewed distribution (more non-IID).

    Args:
        X: Feature array
        y: Target array
        n_clients: Number of clients
        alpha: Dirichlet concentration (0.1-10.0, lower = more non-IID)
        n_classes: Number of classes (auto-detected if None)

    Returns:
        List of (X_client, y_client) tuples

    Reference:
        "Federated Learning with Matched Averaging" (ICLR 2020)
    """
    if n_classes is None:
        n_classes = len(np.unique(y))

    n_samples = len(y)

    # Sample class proportions for each client from Dirichlet
    # shape: (n_clients, n_classes)
    proportions = np.random.dirichlet(alpha * np.ones(n_classes), size=n_clients)

    # Initialize indices for each client and class
    client_indices = [np.zeros(n_classes, dtype=int) for _ in range(n_clients)]
    client_datasets = [[] for _ in range(n_clients)]

    # Assign samples to clients based on class proportions
    for k in range(n_classes):
        # Get indices of samples belonging to class k
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)

        # Split among clients according to proportions
        start_idx = 0
        for i in range(n_clients):
            end_idx = start_idx + int(proportions[i, k] * len(idx_k))

            # Handle edge case where no samples assigned
            if end_idx > start_idx:
                client_datasets[i].extend(idx_k[start_idx:end_idx])
                client_indices[i][k] = end_idx - start_idx

            start_idx = end_idx

    # Convert to arrays
    partitions = []
    for i in range(n_clients):
        if len(client_datasets[i]) == 0:
            logger.warning(f"Client {i} has no samples, skipping")
            continue

        client_idx = np.array(client_datasets[i])
        X_client = X[client_idx]
        y_client = y[client_idx]

        partitions.append((X_client, y_client))

        # Log class distribution
        unique, counts = np.unique(y_client, return_counts=True)
        logger.debug(
            f"Client {i}: {len(X_client)} samples, class distribution: {dict(zip(unique, counts))}"
        )

    logger.info(f"Created Dirichlet partition (alpha={alpha}) across {len(partitions)} clients")
    return partitions


def create_pathological_partition(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    n_classes: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create pathological non-IID partition.

    Each client only has samples from a limited subset of classes.
    Worst-case scenario for federated learning.

    Args:
        X: Feature array
        y: Target array
        n_clients: Number of clients
        n_classes: Number of classes (auto-detected if None)

    Returns:
        List of (X_client, y_client) tuples

    Reference:
        "Federated Learning with Matched Averaging" (ICLR 2020)
    """
    if n_classes is None:
        n_classes = len(np.unique(y))

    # For binary classification, each client gets 1 class
    # For multi-class, each client gets 2 classes (shards)
    if n_classes == 2:
        classes_per_client = 1
    else:
        classes_per_client = 2

    # Sort data by class
    sorted_indices = np.argsort(y)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]

    # Split into shards by class
    shards_per_class = max(1, n_clients // n_classes)
    shard_size = len(y) // (n_classes * shards_per_class)

    partitions = []
    for i in range(n_clients):
        # Determine which classes this client gets
        class_offset = i // (n_clients // n_classes)
        client_classes = [
            (class_offset + c) % n_classes for c in range(classes_per_client)
        ]

        # Get samples for these classes
        client_indices = []
        for cls in client_classes:
            cls_start = cls * shards_per_class * shard_size
            shard_idx = i % shards_per_class
            start = cls_start + shard_idx * shard_size
            end = start + shard_size

            client_indices.extend(range(start, min(end, len(y))))

        if len(client_indices) == 0:
            continue

        client_idx = np.array(client_indices)
        X_client = X_sorted[client_idx]
        y_client = y_sorted[client_idx]

        partitions.append((X_client, y_client))

        unique, counts = np.unique(y_client, return_counts=True)
        logger.debug(
            f"Client {i}: {len(X_client)} samples, classes: {unique.tolist()}"
        )

    logger.info(f"Created pathological partition across {len(partitions)} clients")
    return partitions


def partition_by_account(
    X: np.ndarray,
    y: np.ndarray,
    account_ids: np.ndarray,
    n_clients: int,
    random_seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data by account ID (realistic for fraud detection).

    Each client represents a bank with its own accounts.

    Args:
        X: Feature array
        y: Target array
        account_ids: Account ID for each sample
        n_clients: Number of clients (banks)
        random_seed: Random seed

    Returns:
        List of (X_client, y_client) tuples
    """
    np.random.seed(random_seed)

    unique_accounts = np.unique(account_ids)
    n_accounts = len(unique_accounts)

    # Assign accounts to clients
    account_assignments = np.random.randint(0, n_clients, size=n_accounts)
    account_to_client = dict(zip(unique_accounts, account_assignments))

    # Partition samples by account assignment
    client_datasets = [[] for _ in range(n_clients)]
    for idx, account_id in enumerate(account_ids):
        client_id = account_to_client[account_id]
        client_datasets[client_id].append(idx)

    # Convert to arrays
    partitions = []
    for i in range(n_clients):
        if len(client_datasets[i]) == 0:
            continue

        client_idx = np.array(client_datasets[i])
        X_client = X[client_idx]
        y_client = y[client_idx]

        partitions.append((X_client, y_client))

        fraud_rate = np.mean(y_client) * 100
        logger.info(
            f"Client {i}: {len(X_client)} samples from {len(np.unique(account_ids[client_idx]))} accounts, "
            f"fraud rate: {fraud_rate:.2f}%"
        )

    logger.info(f"Partitioned {n_accounts} accounts across {len(partitions)} clients")
    return partitions


def compute_partition_statistics(
    partitions: List[Tuple[np.ndarray, np.ndarray]],
) -> dict:
    """
    Compute statistics about data partitions.

    Args:
        partitions: List of (X_client, y_client) tuples

    Returns:
        Dictionary of partition statistics
    """
    stats = {
        "n_clients": len(partitions),
        "samples_per_client": [],
        "class_distributions": [],
        "imbalance_ratio": [],
    }

    all_sizes = []
    for X_client, y_client in partitions:
        n_samples = len(X_client)
        all_sizes.append(n_samples)
        stats["samples_per_client"].append(n_samples)

        # Class distribution
        unique, counts = np.unique(y_client, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        stats["class_distributions"].append(class_dist)

        # Imbalance ratio
        if len(counts) > 1:
            ratio = counts.max() / counts.min()
            stats["imbalance_ratio"].append(ratio)

    # Summary statistics
    stats["avg_samples_per_client"] = np.mean(all_sizes)
    stats["std_samples_per_client"] = np.std(all_sizes)
    stats["min_samples_per_client"] = np.min(all_sizes)
    stats["max_samples_per_client"] = np.max(all_sizes)

    if stats["imbalance_ratio"]:
        stats["avg_imbalance_ratio"] = np.mean(stats["imbalance_ratio"])

    return stats


def visualize_partition_stats(
    partitions: List[Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize partition statistics.

    Args:
        partitions: List of (X_client, y_client) tuples
        save_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
        return

    stats = compute_partition_statistics(partitions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Samples per client
    axes[0].bar(range(len(partitions)), stats["samples_per_client"])
    axes[0].set_xlabel("Client ID")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_title("Samples per Client")
    axes[0].axhline(
        stats["avg_samples_per_client"], color="r", linestyle="--", label="Mean"
    )
    axes[0].legend()

    # Class distribution (fraud rate)
    fraud_rates = []
    for class_dist in stats["class_distributions"]:
        total = sum(class_dist.values())
        fraud_count = class_dist.get(1, 0)
        fraud_rates.append(fraud_count / total * 100 if total > 0 else 0)

    axes[1].bar(range(len(fraud_rates)), fraud_rates, color="orange")
    axes[1].set_xlabel("Client ID")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].set_title("Fraud Rate per Client")
    axes[1].axhline(np.mean(fraud_rates), color="r", linestyle="--", label="Mean")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved partition visualization to {save_path}")

    plt.close()
