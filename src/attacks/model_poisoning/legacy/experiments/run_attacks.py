"""
Main experiment orchestrator for model poisoning attacks.

Runs federated learning experiments with various attack strategies
and compares their effectiveness and detectability.
"""

import os
import yaml
import time
import numpy as np
import pandas as pd
import torch
import flwr as fl
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fraud_mlp import FraudMLP
from clients.honest_client import HonestClient
from clients.malicious_client import MaliciousClient
from servers.aggregation import FedAvgWithAttackTracking
from servers.detection import AttackDetector
from utils.metrics import compute_metrics, track_convergence, compute_detectability
from utils.visualization import plot_convergence, plot_detectability, plot_attack_comparison

from attacks import (
    GradientScalingAttack,
    SignFlippingAttack,
    GaussianNoiseAttack,
    TargettedManipulationAttack,
    InnerProductAttack
)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_synthetic_fraud_data(
    num_samples: int = 10000,
    num_features: int = 20,
    fraud_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Generate synthetic credit card fraud dataset.

    Args:
        num_samples: Total number of samples
        num_features: Number of features
        fraud_ratio: Ratio of fraud transactions

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Generate features
    np.random.seed(42)
    X = np.random.randn(num_samples, num_features).astype(np.float32)

    # Generate labels with fraud ratio
    y = np.zeros(num_samples, dtype=np.int64)
    num_fraud = int(num_samples * fraud_ratio)
    fraud_indices = np.random.choice(num_samples, num_fraud, replace=False)
    y[fraud_indices] = 1

    # Split into train/val/test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # Create dataloaders
    batch_size = 32

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def partition_data(
    data_loader: DataLoader,
    num_clients: int
) -> List[DataLoader]:
    """
    Partition data among clients (IID).

    Args:
        data_loader: Original data loader
        num_clients: Number of clients

    Returns:
        List of client data loaders
    """
    all_data = []
    all_targets = []

    for data, targets in data_loader:
        all_data.append(data)
        all_targets.append(targets)

    all_data = torch.cat(all_data)
    all_targets = torch.cat(all_targets)

    # Shuffle and partition
    indices = np.random.permutation(len(all_data))
    shard_size = len(all_data) // num_clients

    client_loaders = []

    for i in range(num_clients):
        start_idx = i * shard_size
        end_idx = start_idx + shard_size if i < num_clients - 1 else len(all_data)

        client_data = all_data[indices[start_idx:end_idx]]
        client_targets = all_targets[indices[start_idx:end_idx]]

        client_dataset = TensorDataset(client_data, client_targets)
        client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)
        client_loaders.append(client_loader)

    return client_loaders


def run_baseline(
    num_rounds: int = 50,
    num_clients: int = 10,
    attacker_fraction: float = 0.0
) -> Dict:
    """
    Run baseline federated learning without attacks.

    Args:
        num_rounds: Number of training rounds
        num_clients: Total number of clients
        attacker_fraction: Fraction of malicious clients (0.0 for baseline)

    Returns:
        Dictionary with training history and metrics
    """
    print(f"Running baseline training (no attacks)...")

    # Load config
    fl_config = load_config("config/fl_config.yaml")

    # Generate data
    train_loader, val_loader, test_loader = generate_synthetic_fraud_data()
    client_loaders = partition_data(train_loader, num_clients)

    # Initialize model
    model = FraudMLP(
        input_dim=fl_config["input_dim"],
        hidden_dims=fl_config["hidden_dims"],
        output_dim=fl_config["output_dim"]
    )

    # Create clients
    clients = []
    for i in range(num_clients):
        client = HonestClient(
            model=model,
            train_loader=client_loaders[i],
            test_loader=test_loader,
            client_id=i,
            device="cpu"
        )
        clients.append(client)

    # Initialize server
    initial_params = [param.data.cpu().numpy() for param in model.parameters()]
    server = FedAvgWithAttackTracking(
        initial_parameters=initial_params,
        detect_attacks=True
    )

    # Training loop
    metrics_history = []

    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")

        # Select clients
        num_selected = max(fl_config["min_fit_clients"], int(num_clients * fl_config["client_fraction"]))
        selected_indices = np.random.choice(num_clients, num_selected, replace=False)

        # Client training
        client_results = []
        for idx in selected_indices:
            client = clients[idx]

            params = server.get_current_parameters()
            config = {
                "local_epochs": fl_config["local_epochs"],
                "lr": fl_config["learning_rate"],
                "server_round": round_num
            }

            updated_params, num_examples, metrics = client.fit(params, config)

            # Create FitRes-like object
            from flwr.common import FitRes, Parameters, Metrics
            from flwr.common.typing import Scalar

            fit_res = FitRes(
                parameters=Parameters(tensors=[p.tobytes() for p in updated_params]),
                num_examples=num_examples,
                metrics=metrics
            )
            client_results.append((idx, fit_res))

        # Aggregate
        aggregated_params, agg_metrics = server.aggregate_fit(client_results)

        # Evaluate
        model_params = [torch.from_numpy(p) for p in aggregated_params]
        for param, model_param in zip(model.parameters(), model_params):
            param.data = model_param

        metrics = compute_metrics(model, test_loader, device="cpu")
        metrics["round"] = round_num + 1
        metrics_history.append(metrics)

        print(f"  Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")

    convergence_info = track_convergence(metrics_history)

    return {
        "metrics_history": metrics_history,
        "convergence": convergence_info,
        "update_history": server.get_update_history()
    }


def run_single_attack(
    attack_name: str,
    attack_params: Dict,
    num_rounds: int = 50,
    num_clients: int = 10,
    attacker_fraction: float = 0.2
) -> Dict:
    """
    Run federated learning with a specific attack.

    Args:
        attack_name: Name of the attack strategy
        attack_params: Parameters for the attack
        num_rounds: Number of training rounds
        num_clients: Total number of clients
        attacker_fraction: Fraction of malicious clients

    Returns:
        Dictionary with training history and attack impact
    """
    print(f"\nRunning attack: {attack_name}")
    print(f"Attacker fraction: {attacker_fraction}")

    # Load config
    fl_config = load_config("config/fl_config.yaml")

    # Generate data
    train_loader, val_loader, test_loader = generate_synthetic_fraud_data()
    client_loaders = partition_data(train_loader, num_clients)

    # Initialize model
    model = FraudMLP(
        input_dim=fl_config["input_dim"],
        hidden_dims=fl_config["hidden_dims"],
        output_dim=fl_config["output_dim"]
    )

    # Create attack strategy
    if attack_name == "gradient_scaling":
        attack_strategy = GradientScalingAttack(
            scaling_factor=attack_params.get("scaling_factor", 10.0)
        )
    elif attack_name == "sign_flipping":
        attack_strategy = SignFlippingAttack(
            factor=attack_params.get("factor", -1.0)
        )
    elif attack_name == "gaussian_noise":
        attack_strategy = GaussianNoiseAttack(
            noise_std=attack_params.get("noise_std", 0.5)
        )
    elif attack_name == "targetted_manipulation":
        attack_strategy = TargettedManipulationAttack(
            target_layers=attack_params.get("target_layers", ["fc2.weight", "fc2.bias"]),
            perturbation_scale=attack_params.get("perturbation_scale", 5.0)
        )
    elif attack_name == "inner_product":
        attack_strategy = InnerProductAttack(
            optimization_steps=attack_params.get("optimization_steps", 10),
            step_size=attack_params.get("step_size", 0.1)
        )
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    # Create clients
    num_attackers = int(num_clients * attacker_fraction)
    clients = []

    for i in range(num_clients):
        if i < num_attackers:
            # Malicious client
            client = MaliciousClient(
                model=model,
                train_loader=client_loaders[i],
                test_loader=test_loader,
                client_id=i,
                attack_strategy=attack_strategy,
                attack_timing=attack_params.get("timing", "continuous"),
                device="cpu"
            )
        else:
            # Honest client
            client = HonestClient(
                model=model,
                train_loader=client_loaders[i],
                test_loader=test_loader,
                client_id=i,
                device="cpu"
            )
        clients.append(client)

    # Initialize server
    initial_params = [param.data.cpu().numpy() for param in model.parameters()]
    server = FedAvgWithAttackTracking(
        initial_parameters=initial_params,
        detect_attacks=True
    )

    # Initialize attack detector
    detector = AttackDetector(
        l2_norm_threshold=10.0,
        cosine_similarity_threshold=-0.5
    )

    # Training loop
    metrics_history = []
    detection_history = []
    round_times = []

    for round_num in range(num_rounds):
        start_time = time.time()
        print(f"Round {round_num + 1}/{num_rounds}")

        # Select clients
        num_selected = max(fl_config["min_fit_clients"], int(num_clients * fl_config["client_fraction"]))
        selected_indices = np.random.choice(num_clients, num_selected, replace=False)

        # Client training
        client_results = []
        client_updates = []
        client_ids = []

        for idx in selected_indices:
            client = clients[idx]

            params = server.get_current_parameters()
            config = {
                "local_epochs": fl_config["local_epochs"],
                "lr": fl_config["learning_rate"],
                "server_round": round_num
            }

            updated_params, num_examples, metrics = client.fit(params, config)

            # Store for detection
            flat_update = np.concatenate([p.flatten() for p in updated_params])
            client_updates.append(flat_update)
            client_ids.append(idx)

            # Create FitRes-like object
            from flwr.common import FitRes, Parameters

            fit_res = FitRes(
                parameters=Parameters(tensors=[p.tobytes() for p in updated_params]),
                num_examples=num_examples,
                metrics=metrics
            )
            client_results.append((idx, fit_res))

        # Detect anomalies
        detection_result = detector.detect_anomalies(client_updates, client_ids)
        detection_history.append(detection_result)

        # Aggregate
        aggregated_params, agg_metrics = server.aggregate_fit(client_results)

        # Evaluate
        model_params = [torch.from_numpy(p) for p in aggregated_params]
        for param, model_param in zip(model.parameters(), model_params):
            param.data = model_param

        metrics = compute_metrics(model, test_loader, device="cpu")
        metrics["round"] = round_num + 1
        metrics_history.append(metrics)

        round_time = time.time() - start_time
        round_times.append(round_time)

        print(f"  Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")
        print(f"  Detected suspicious: {detection_result['suspicious_clients']}")

    convergence_info = track_convergence(metrics_history)
    detectability_info = compute_detectability(
        server.get_update_history(),
        detection_history
    )

    return {
        "attack_name": attack_name,
        "attack_params": attack_params,
        "metrics_history": metrics_history,
        "convergence": convergence_info,
        "detectability": detectability_info,
        "update_history": server.get_update_history(),
        "detection_history": detection_history,
        "avg_round_time": np.mean(round_times)
    }


def compare_all_attacks(
    num_rounds: int = 50,
    num_clients: int = 10,
    attacker_fraction: float = 0.2,
    output_dir: str = "results/logs"
) -> pd.DataFrame:
    """
    Compare all attack strategies with fair comparison.

    Args:
        num_rounds: Number of training rounds
        num_clients: Total number of clients
        attacker_fraction: Fraction of malicious clients
        output_dir: Directory to save results

    Returns:
        DataFrame with comparison results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MODEL POISONING ATTACK EXPERIMENT")
    print("=" * 60)
    print(f"Rounds: {num_rounds}")
    print(f"Clients: {num_clients}")
    print(f"Attacker fraction: {attacker_fraction}")
    print("=" * 60)

    # Run baseline first
    baseline_results = run_baseline(
        num_rounds=num_rounds,
        num_clients=num_clients,
        attacker_fraction=0.0
    )

    # Define attacks to test
    attacks = [
        {
            "name": "gradient_scaling",
            "params": {"scaling_factor": 10.0}
        },
        {
            "name": "sign_flipping",
            "params": {"factor": -1.0}
        },
        {
            "name": "gaussian_noise",
            "params": {"noise_std": 0.5}
        },
        {
            "name": "targetted_manipulation",
            "params": {"perturbation_scale": 5.0}
        },
        {
            "name": "inner_product",
            "params": {"optimization_steps": 10}
        }
    ]

    results = []
    all_metrics_history = {"baseline": baseline_results["metrics_history"]}

    # Run each attack
    for attack_config in attacks:
        attack_results = run_single_attack(
            attack_name=attack_config["name"],
            attack_params=attack_config["params"],
            num_rounds=num_rounds,
            num_clients=num_clients,
            attacker_fraction=attacker_fraction
        )

        all_metrics_history[attack_config["name"]] = attack_results["metrics_history"]

        # Compile results
        result_row = {
            "attack": attack_config["name"],
            "final_accuracy": attack_results["convergence"]["final_accuracy"],
            "convergence_round": attack_results["convergence"]["convergence_round"],
            "detection_rate": attack_results["detectability"]["detection_rate"],
            "false_positive_rate": attack_results["detectability"]["false_positive_rate"],
            "train_time_avg": attack_results["avg_round_time"]
        }
        results.append(result_row)

    # Add baseline
    results.insert(0, {
        "attack": "baseline",
        "final_accuracy": baseline_results["convergence"]["final_accuracy"],
        "convergence_round": baseline_results["convergence"]["convergence_round"],
        "detection_rate": 0.0,
        "false_positive_rate": 0.0,
        "train_time_avg": 0.0  # Will compute if needed
    })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)

    # Save results
    comparison_df.to_csv(f"{output_dir}/attack_comparison.csv", index=False)
    print(f"\nSaved comparison results to {output_dir}/attack_comparison.csv")

    # Generate plots
    print("\nGenerating plots...")
    plot_convergence(all_metrics_history, f"{output_dir}/convergence_curves.png")
    plot_detectability(
        {r["attack"]: {"detection_rate": r["detection_rate"], "false_positive_rate": r["false_positive_rate"]}
         for r in results if r["attack"] != "baseline"},
        f"{output_dir}/detectability_analysis.png"
    )
    plot_attack_comparison(comparison_df, f"{output_dir}/attack_comparison.png")

    # Print summary
    print("\n" + "=" * 60)
    print("ATTACK COMPARISON SUMMARY")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)

    return comparison_df


if __name__ == "__main__":
    # Run full comparison
    results = compare_all_attacks(
        num_rounds=50,
        num_clients=10,
        attacker_fraction=0.2
    )
