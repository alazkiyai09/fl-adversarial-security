"""
Experiment orchestrator for label flipping attacks on Federated Learning.

This module provides functions to run federated learning experiments with
label flipping attacks, measuring their impact on model performance.
"""

import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fraud_mlp import FraudMLP, set_model_parameters, get_model_parameters
from clients.honest_client import create_honest_client, HonestClient
from clients.malicious_client import create_malicious_client, MaliciousClient
from servers.attack_server import AttackMetricsStrategy, create_eval_fn
from config.attack_config import AttackConfig, get_attack_configs
from utils.data_loader import FraudDataLoader
from utils.poisoning_utils import select_malicious_clients
from metrics.attack_metrics import compare_histories, calculate_robustness_metrics
from metrics.visualization import (
    plot_accuracy_over_rounds,
    plot_per_class_accuracy,
    plot_attacker_fraction_impact,
    plot_attack_type_comparison,
    plot_convergence_comparison,
    create_summary_report,
)


@dataclass
class ExperimentResult:
    """
    Container for experiment results.

    Attributes:
        attack_config: Attack configuration used
        history: Training history (metrics over rounds)
        final_metrics: Final metrics summary
    """
    attack_config: AttackConfig
    history: Dict[str, List] = field(default_factory=dict)
    final_metrics: Dict = field(default_factory=dict)


def run_single_experiment(
    attack_config: AttackConfig,
    data: Dict,
    num_rounds: int = 100,
    num_clients: int = 10,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    device: str = "cpu",
    server_address: str = "127.0.0.1:8080"
) -> ExperimentResult:
    """
    Run a single federated learning experiment with label flipping attack.

    Args:
        attack_config: Attack configuration
        data: Data dictionary containing train_loaders, val_loader, test_loader
        num_rounds: Number of training rounds
        num_clients: Total number of clients
        local_epochs: Number of local epochs per client
        learning_rate: Learning rate for clients
        device: Device to run on
        server_address: Server address for Flower

    Returns:
        ExperimentResult containing history and metrics
    """
    print(f"\n{'='*70}")
    print(f"Running experiment: {attack_config.attack_type} attack")
    print(f"  Flip rate: {attack_config.flip_rate}")
    print(f"  Malicious fraction: {attack_config.malicious_fraction}")
    print(f"  Attack start round: {attack_config.attack_start_round}")
    print(f"{'='*70}\n")

    # Select malicious clients
    malicious_indices = select_malicious_clients(
        total_clients=num_clients,
        malicious_indices=attack_config.malicious_client_indices,
        malicious_fraction=attack_config.malicious_fraction,
        seed=attack_config.random_seed
    )

    print(f"Malicious clients: {malicious_indices}")

    # Create client function
    def client_fn(cid: str) -> fl.client.NumPyClient:
        """Create a client (honest or malicious)."""
        client_id = int(cid)

        if client_id in malicious_indices:
            # Create malicious client
            return create_malicious_client(
                train_loader=data["train_loaders"][client_id],
                test_loader=data["val_loader"],
                client_id=str(client_id),
                attack_config=attack_config,
                input_size=30,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                device=device
            )
        else:
            # Create honest client
            return create_honest_client(
                train_loader=data["train_loaders"][client_id],
                test_loader=data["val_loader"],
                client_id=str(client_id),
                input_size=30,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                device=device
            )

    # Create evaluation function for server
    eval_fn = create_eval_fn(
        test_loader=data["test_loader"],
        input_size=30,
        device=device
    )

    # Initialize model parameters
    init_model = FraudMLP(input_size=30)
    init_params = get_model_parameters(init_model)

    # Create server strategy with metrics tracking
    strategy = AttackMetricsStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        eval_fn=eval_fn,
        initial_parameters=init_params,
    )

    # Start simulation
    print(f"Starting FL simulation with {num_clients} clients for {num_rounds} rounds...")

    # Use Flower's simulation API
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if device == "cpu" else 1.0},
    )

    # Extract metrics from strategy
    metrics_history = strategy.get_history()

    # Create experiment result
    result = ExperimentResult(
        attack_config=attack_config,
        history=metrics_history,
        final_metrics=strategy.get_convergence_summary()
    )

    print(f"\nExperiment completed!")
    print(f"  Final accuracy: {metrics_history.get('global_accuracy', [None])[-1]:.4f}")
    print(f"  Final fraud accuracy: {metrics_history.get('accuracy_fraud', [None])[-1]:.4f}")
    print(f"  Final legitimate accuracy: {metrics_history.get('accuracy_legitimate', [None])[-1]:.4f}")

    return result


def run_baseline_experiment(
    data: Dict,
    num_rounds: int = 100,
    num_clients: int = 10,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    device: str = "cpu"
) -> ExperimentResult:
    """
    Run a baseline experiment without any attack.

    Args:
        data: Data dictionary
        num_rounds: Number of training rounds
        num_clients: Total number of clients
        local_epochs: Number of local epochs per client
        learning_rate: Learning rate for clients
        device: Device to run on

    Returns:
        ExperimentResult for baseline
    """
    print(f"\n{'='*70}")
    print("Running BASELINE experiment (no attack)")
    print(f"{'='*70}\n")

    # Create baseline config with no attack
    baseline_config = AttackConfig(
        attack_type="random",
        flip_rate=0.0,
        malicious_fraction=0.0,
        random_seed=42
    )

    return run_single_experiment(
        attack_config=baseline_config,
        data=data,
        num_rounds=num_rounds,
        num_clients=num_clients,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        device=device
    )


def run_attacker_fraction_sweep(
    data: Dict,
    fractions: List[float] = [0.1, 0.2, 0.3, 0.5],
    num_rounds: int = 100,
    num_clients: int = 10,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    device: str = "cpu",
    attack_type: str = "random",
    flip_rate: float = 0.3
) -> Dict[float, ExperimentResult]:
    """
    Run experiments across different attacker fractions.

    Args:
        data: Data dictionary
        fractions: List of attacker fractions to test
        num_rounds: Number of training rounds
        num_clients: Total number of clients
        local_epochs: Number of local epochs per client
        learning_rate: Learning rate for clients
        device: Device to run on
        attack_type: Type of attack to use
        flip_rate: Flip rate for the attack

    Returns:
        Dictionary mapping attacker fraction to ExperimentResult
    """
    print(f"\n{'='*70}")
    print(f"Running attacker fraction sweep: {fractions}")
    print(f"  Attack type: {attack_type}")
    print(f"  Flip rate: {flip_rate}")
    print(f"{'='*70}\n")

    results = {}

    for fraction in fractions:
        config = AttackConfig(
            attack_type=attack_type,
            flip_rate=flip_rate,
            malicious_fraction=fraction,
            random_seed=42
        )

        result = run_single_experiment(
            attack_config=config,
            data=data,
            num_rounds=num_rounds,
            num_clients=num_clients,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=device
        )

        results[fraction] = result

    return results


def run_attack_type_comparison(
    data: Dict,
    attack_types: List[str] = ["random", "targeted", "inverse"],
    num_rounds: int = 100,
    num_clients: int = 10,
    malicious_fraction: float = 0.2,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    device: str = "cpu"
) -> Dict[str, ExperimentResult]:
    """
    Run experiments comparing different attack types.

    Args:
        data: Data dictionary
        attack_types: List of attack types to compare
        num_rounds: Number of training rounds
        num_clients: Total number of clients
        malicious_fraction: Fraction of malicious clients
        local_epochs: Number of local epochs per client
        learning_rate: Learning rate for clients
        device: Device to run on

    Returns:
        Dictionary mapping attack type to ExperimentResult
    """
    print(f"\n{'='*70}")
    print(f"Running attack type comparison: {attack_types}")
    print(f"  Malicious fraction: {malicious_fraction}")
    print(f"{'='*70}\n")

    results = {}

    for attack_type in attack_types:
        # Set appropriate flip rate based on attack type
        if attack_type == "inverse":
            flip_rate = 1.0
        elif attack_type == "targeted":
            flip_rate = 0.5
        else:  # random
            flip_rate = 0.3

        config = AttackConfig(
            attack_type=attack_type,
            flip_rate=flip_rate,
            malicious_fraction=malicious_fraction,
            random_seed=42
        )

        result = run_single_experiment(
            attack_config=config,
            data=data,
            num_rounds=num_rounds,
            num_clients=num_clients,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=device
        )

        results[attack_type] = result

    return results


def generate_all_visualizations(
    baseline_result: ExperimentResult,
    attack_results: Dict[str, ExperimentResult],
    output_dir: str = "results/figures"
) -> None:
    """
    Generate all visualization plots for experiment results.

    Args:
        baseline_result: Baseline experiment result
        attack_results: Dictionary of attack experiment results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Generating visualizations...")
    print(f"{'='*70}\n")

    # Plot accuracy over rounds for each attack
    for attack_name, result in attack_results.items():
        if isinstance(attack_name, str):
            plot_accuracy_over_rounds(
                baseline_result.history,
                result.history,
                attack_name=f"{attack_name} Attack",
                save_path=output_path / f"accuracy_{attack_name}.png",
                show=False
            )

    # Plot per-class accuracy
    for attack_name, result in attack_results.items():
        if isinstance(attack_name, str):
            plot_per_class_accuracy(
                baseline_result.history,
                result.history,
                attack_name=f"{attack_name} Attack",
                save_path=output_path / f"per_class_{attack_name}.png",
                show=False
            )

    # Plot attacker fraction impact
    if all(isinstance(k, float) for k in attack_results.keys()):
        baseline_acc = baseline_result.history.get("global_accuracy", [None])[-1]
        if baseline_acc is not None:
            plot_attacker_fraction_impact(
                {k: v.history for k, v in attack_results.items()},
                baseline_acc,
                save_path=output_path / "attacker_fraction_impact.png",
                show=False
            )

    # Plot attack type comparison
    if all(isinstance(k, str) for k in attack_results.keys()):
        baseline_acc = baseline_result.history.get("global_accuracy", [None])[-1]
        if baseline_acc is not None:
            plot_attack_type_comparison(
                {k: v.history for k, v in attack_results.items()},
                baseline_acc,
                save_path=output_path / "attack_type_comparison.png",
                show=False
            )

    print(f"Saved all visualizations to {output_dir}")


def main():
    """
    Main entry point for running experiments.

    Run all experiments and generate visualizations.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run label flipping attack experiments")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of training rounds")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local epochs per client")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--output-dir", type=str, default="results/figures", help="Output directory for plots")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    data_loader = FraudDataLoader()
    data = data_loader.load_and_prepare(
        num_clients=args.num_clients,
        batch_size=32,
        partition_type="iid"
    )

    # Run baseline
    baseline = run_baseline_experiment(
        data=data,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        device=args.device
    )

    # Run attacker fraction sweep
    fraction_results = run_attacker_fraction_sweep(
        data=data,
        fractions=[0.1, 0.2, 0.3, 0.5],
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        device=args.device
    )

    # Run attack type comparison
    attack_type_results = run_attack_type_comparison(
        data=data,
        attack_types=["random", "targeted", "inverse"],
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        device=args.device
    )

    # Generate visualizations
    generate_all_visualizations(
        baseline_result=baseline,
        attack_results={"random": fraction_results[0.2]},
        output_dir=args.output_dir
    )

    print("\n" + "="*70)
    print("All experiments completed!")
    print("="*70)


if __name__ == "__main__":
    main()
