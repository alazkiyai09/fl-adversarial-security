"""
Main experiment runner for defense comparison.

Compares FoolsGold against other defenses (FedAvg, Krum, TrimmedMean, etc.)
under various attack scenarios.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import Parameters, ndarrays_to_parameters
import matplotlib.pyplot as plt

from ..aggregators import (
    FoolsGoldAggregator,
    KrumAggregator,
    MultiKrumAggregator,
    TrimmedMeanAggregator,
)
from ..attacks import SybilAttack, CollusionAttack
from ..clients import create_client
from ..server import create_foolsgold_server
from ..utils.metrics import DefenseMetrics, compute_accuracy, compute_similarity_metrics


def generate_synthetic_data(
    num_samples: int = 1000,
    num_features: int = 20,
    fraud_ratio: float = 0.1,
    num_clients: int = 10,
    samples_per_client: int = 100,
    random_seed: Optional[int] = None
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Generate synthetic fraud detection data.

    Args:
        num_samples: Total number of samples
        num_features: Number of features
        fraud_ratio: Fraction of fraudulent samples
        num_clients: Number of clients
        samples_per_client: Samples per client
        random_seed: Random seed

    Returns:
        Tuple of (train_loaders, test_loaders)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Generate features
    X = np.random.randn(num_samples, num_features).astype(np.float32)

    # Generate labels (fraud = 1, legit = 0)
    y = np.zeros(num_samples, dtype=np.int64)
    num_fraud = int(num_samples * fraud_ratio)
    fraud_indices = np.random.choice(num_samples, num_fraud, replace=False)
    y[fraud_indices] = 1

    # Convert to tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # Split into train and test
    split = int(0.8 * num_samples)
    X_train, X_test = X_tensor[:split], X_tensor[split:]
    y_train, y_test = y_tensor[:split], y_tensor[split:]

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Split among clients
    train_loaders = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = min(start_idx + samples_per_client, len(train_dataset))

        if start_idx < len(train_dataset):
            subset = torch.utils.data.Subset(train_dataset, range(start_idx, end_idx))
            loader = DataLoader(subset, batch_size=32, shuffle=True)
            train_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_loaders = [test_loader] * num_clients

    return train_loaders, test_loaders


def run_single_experiment(
    defense: str,
    attack_type: str,
    num_malicious: int,
    num_clients: int = 10,
    num_rounds: int = 50,
    num_features: int = 20,
    random_seed: Optional[int] = None
) -> Dict[str, List[float]]:
    """
    Run a single FL experiment with specific defense and attack.

    Args:
        defense: Defense type ('foolsgold', 'krum', 'multi_krum', 'trimmed_mean', 'fedavg')
        attack_type: Attack type ('sybil', 'collusion', 'none')
        num_malicious: Number of malicious clients
        num_clients: Total number of clients
        num_rounds: Number of training rounds
        num_features: Number of input features
        random_seed: Random seed

    Returns:
        Dictionary of metrics over rounds
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    print(f"\n{'='*60}")
    print(f"Experiment: {defense} vs {attack_type} ({num_malicious} malicious)")
    print(f"{'='*60}")

    # Generate data
    train_loaders, test_loaders = generate_synthetic_data(
        num_clients=num_clients,
        num_features=num_features,
        random_seed=random_seed
    )

    # Initialize global model
    from ..models.fraud_net import FraudNet, get_model_parameters
    global_model = FraudNet(input_dim=num_features)
    global_params = ndarrays_to_parameters(get_model_parameters(global_model))

    # Create aggregator based on defense
    if defense == "foolsgold":
        aggregator = FoolsGoldAggregator(
            history_length=10,
            similarity_threshold=0.9,
            lr_scale_factor=0.1
        )
    elif defense == "krum":
        aggregator = KrumAggregator(num_malicious=num_malicious)
    elif defense == "multi_krum":
        aggregator = MultiKrumAggregator(num_malicious=num_malicious)
    elif defense == "trimmed_mean":
        aggregator = TrimmedMeanAggregator(trim_ratio=0.1)
    elif defense == "fedavg":
        # Simple average (no defense)
        aggregator = TrimmedMeanAggregator(trim_ratio=0.0)
    else:
        raise ValueError(f"Unknown defense: {defense}")

    # Create attack
    malicious_ids = list(range(num_clients - num_malicious, num_clients))

    # Metrics tracker
    metrics_tracker = DefenseMetrics()

    # Training rounds
    for round_num in range(num_rounds):
        # Simulate client training
        results = []
        contribution_scores = None

        for client_id in range(num_clients):
            is_malicious = client_id in malicious_ids

            # Create client
            client = create_client(
                client_id=client_id,
                train_loader=train_loaders[client_id],
                test_loader=test_loaders[client_id],
                input_dim=num_features,
                num_epochs=1,
                is_malicious=is_malicious,
                attack_type="sign_flip" if is_malicious and attack_type != "none" else None
            )

            # Train client
            fit_res, _, _ = client.fit(global_params, {})

            # Apply Sybil attack (coordinate updates)
            if is_malicious and attack_type == "sybil":
                # For Sybil, make malicious clients more similar
                pass  # Already handled by client attack_type

            results.append((None, fit_res))  # ClientProxy placeholder

        # Aggregate
        global_params, agg_metrics = aggregator.aggregate(results)

        # Get contribution scores if FoolsGold
        if defense == "foolsgold":
            history = aggregator.get_metrics()
            if history.get("contribution_scores"):
                contribution_scores = history["contribution_scores"][-1]

        # Evaluate
        global_model_eval = FraudNet(input_dim=num_features)
        from ..models.fraud_net import set_model_parameters
        set_model_parameters(global_model_eval, global_params.tensors)

        accuracy = compute_accuracy(global_model_eval, test_loaders[0])

        # Get flagged Sybils if FoolsGold
        flagged_sybils = []
        if defense == "foolsgold":
            history = aggregator.get_metrics()
            if history.get("flagged_sybils"):
                flagged_sybils = history["flagged_sybils"][-1]

        # Track metrics
        metrics_tracker.add_round(
            round_num=round_num,
            accuracy=accuracy,
            loss=agg_metrics.get("avg_loss", 0.0),
            attack_success_rate=0.0,  # Could compute if needed
            contribution_scores=contribution_scores,
            flagged_sybils=flagged_sybils,
            malicious_ids=malicious_ids if attack_type != "none" else []
        )

        if (round_num + 1) % 10 == 0:
            print(f"Round {round_num + 1}/{num_rounds}: Accuracy = {accuracy:.4f}")

    return metrics_tracker.get_final_metrics()


def run_defense_comparison(
    attacks: List[str] = ["sybil", "collusion", "none"],
    defenses: List[str] = ["foolsgold", "krum", "multi_krum", "trimmed_mean", "fedavg"],
    num_malicious_values: List[int] = [1, 2, 3],
    num_rounds: int = 50,
    output_dir: str = "results"
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Run comprehensive defense comparison across all attacks and defenses.

    Args:
        attacks: List of attack types to test
        defenses: List of defenses to compare
        num_malicious_values: Number of malicious clients to test
        num_rounds: Training rounds per experiment
        output_dir: Directory to save results

    Returns:
        Nested dictionary of results
    """
    all_results = {}

    for attack in attacks:
        all_results[attack] = {}

        for defense in defenses:
            all_results[attack][defense] = {}

            for num_malicious in num_malicious_values:
                print(f"\n{'='*80}")
                print(f"Running: {attack} attack with {num_malicious} malicious clients vs {defense}")
                print(f"{'='*80}")

                metrics = run_single_experiment(
                    defense=defense,
                    attack_type=attack,
                    num_malicious=num_malicious,
                    num_rounds=num_rounds
                )

                all_results[attack][defense][num_malicious] = metrics

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "defense_comparison.json")

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return all_results


def plot_defense_comparison(
    results: Dict,
    output_dir: str = "results/figures"
) -> None:
    """
    Generate comparison plots.

    Args:
        results: Results dictionary from run_defense_comparison
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Final accuracy comparison
    attacks = list(results.keys())
    defenses = list(next(iter(results.values())).keys())

    for attack in attacks:
        fig, ax = plt.subplots(figsize=(10, 6))

        for defense in defenses:
            accuracies = []
            num_malicious_list = sorted(results[attack][defense].keys())

            for num_mal in num_malicious_list:
                metrics = results[attack][defense][num_mal]
                accuracies.append(metrics.get("final_accuracy", 0.0))

            ax.plot(num_malicious_list, accuracies, marker="o", label=defense)

        ax.set_xlabel("Number of Malicious Clients")
        ax.set_ylabel("Final Accuracy")
        ax.set_title(f"Defense Comparison: {attack.upper()} Attack")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"accuracy_{attack}.png"), dpi=300)
        plt.close()

    print(f"Plots saved to {output_dir}")
