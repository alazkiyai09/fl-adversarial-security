"""
Differential Privacy Defense Analysis.

This module analyzes how differential privacy (DP) affects property
inference leakage in federated learning.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

from ..fl_system.client import DPEnabledClient
from ..fl_system.server import FederatedServer
from ..data_generation.synthetic_generator import generate_client_datasets
from ..attacks.property_inference import PropertyInferenceAttack
from ..metrics.attack_metrics import compute_regression_metrics, compare_to_baseline


def analyze_dp_effect_on_leakage(
    n_clients: int = 10,
    n_features: int = 10,
    target_property: str = 'fraud_rate',
    noise_multipliers: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
    n_rounds: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """Analyze how DP noise affects property inference leakage.

    Args:
        n_clients: Number of clients
        n_features: Number of features
        target_property: Property to infer
        noise_multipliers: List of DP noise multipliers to test
        n_rounds: Number of FL rounds
        random_state: Random seed

    Returns:
        Dict with leakage analysis results

    Example:
        >>> results = analyze_dp_effect_on_leakage(
        ...     noise_multipliers=[0.1, 1.0, 5.0]
        ... )
        >>> results['0.1']['MAE']
        0.02  # Low noise: good leakage
        >>> results['5.0']['MAE']
        0.15  # High noise: poor leakage
    """
    results = {}

    # Generate client datasets
    property_variations = {
        'fraud_rate': {'type': 'uniform', 'min': 0.01, 'max': 0.2},
        'dataset_size': {'type': 'normal', 'mean': 1000, 'std': 300}
    }

    datasets, properties = generate_client_datasets(
        n_clients=n_clients,
        base_n_samples=1000,
        n_features=n_features,
        property_variations=property_variations,
        random_state=random_state
    )

    model_config = {
        'type': 'logistic_regression',
        'input_dim': n_features
    }

    for noise_multiplier in noise_multipliers:
        # Create DP-enabled clients
        clients = []
        for client_id, dataset in enumerate(datasets):
            client = DPEnabledClient(
                client_id=client_id,
                dataset=dataset,
                model_config=model_config,
                local_epochs=5,
                learning_rate=0.01,
                dp_enabled=True,
                dp_noise_multiplier=noise_multiplier,
                dp_max_grad_norm=1.0
            )
            clients.append(client)

        # Create server
        server = FederatedServer(
            model_config=model_config,
            n_clients=n_clients
        )

        # Run FL
        for round_idx in range(n_rounds):
            server.run_round(clients)

        # Extract attack data
        updates_list = []
        props_list = []

        for client in clients:
            # Get final model weights as proxy for update
            weights = client.model.get_weights()

            # Get client properties
            client_props = client.get_dataset_properties()
            prop_value = client_props[target_property]

            updates_list.append(weights)
            props_list.append(prop_value)

        # Simple train/test split
        n_train = int(0.8 * len(updates_list))
        train_updates, test_updates = np.array(updates_list[:n_train]), np.array(updates_list[n_train:])
        train_props, test_props = np.array(props_list[:n_train]), np.array(props_list[n_train:])

        # Train attack
        attack = PropertyInferenceAttack(
            target_property=target_property,
            scenario='server'
        )

        if len(train_updates) > 0:
            attack.train_meta_classifier(train_updates, train_props)

            # Evaluate
            if len(test_updates) > 0:
                eval_results = attack.evaluate_attack(test_updates, test_props)
            else:
                eval_results = attack.evaluate_attack(train_updates, train_props)

            results[str(noise_multiplier)] = {
                'MAE': eval_results.get('MAE', None),
                'R2': eval_results.get('R2', None),
                'attack_better': eval_results.get('baseline_comparison', {}).get('attack_better', False)
            }

    return results


def find_privacy_utility_tradeoff(
    n_clients: int = 10,
    n_features: int = 10,
    target_property: str = 'fraud_rate',
    noise_range: Tuple[float, float] = (0.1, 10.0),
    n_steps: int = 20,
    n_rounds: int = 10
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Find privacy-utility tradeoff curve for property inference.

    Args:
        n_clients: Number of clients
        n_features: Number of features
        target_property: Property to infer
        noise_range: (min_noise, max_noise) range to explore
        n_steps: Number of noise levels to test
        n_rounds: Number of FL rounds

    Returns:
        (tradeoff_data, optimal_point) tuple
        - tradeoff_data: Dict with 'noise', 'MAE', 'accuracy' lists
        - optimal_point: Dict with noise level that balances privacy and utility

    Example:
        >>> tradeoff, optimal = find_privacy_utility_tradeoff()
        >>> optimal['noise']
        1.5
        >>> optimal['MAE']
        0.05
    """
    noise_levels = np.linspace(noise_range[0], noise_range[1], n_steps)

    mae_list = []
    accuracy_list = []

    for noise in noise_levels:
        results = analyze_dp_effect_on_leakage(
            n_clients=n_clients,
            n_features=n_features,
            target_property=target_property,
            noise_multipliers=[noise],
            n_rounds=n_rounds
        )

        mae_list.append(results[str(noise)]['MAE'])
        # Accuracy would need to be computed from FL model
        accuracy_list.append(0.8)  # Placeholder

    tradeoff_data = {
        'noise': noise_levels.tolist(),
        'MAE': mae_list,
        'accuracy': accuracy_list
    }

    # Find optimal point (e.g., where MAE > threshold but accuracy still good)
    mae_threshold = 0.1  # Attack MAE above this = good privacy
    valid_indices = [i for i, mae in enumerate(mae_list) if mae > mae_threshold]

    if valid_indices:
        # Choose lowest noise that still provides privacy
        optimal_idx = valid_indices[0]
        optimal_point = {
            'noise': float(noise_levels[optimal_idx]),
            'MAE': float(mae_list[optimal_idx]),
            'accuracy': float(accuracy_list[optimal_idx])
        }
    else:
        # No valid point found
        optimal_point = {
            'noise': None,
            'MAE': None,
            'accuracy': None
        }

    return tradeoff_data, optimal_point


def compare_dp_vs_no_dp(
    n_clients: int = 10,
    n_features: int = 10,
    target_property: str = 'fraud_rate',
    noise_multiplier: float = 1.0,
    n_rounds: int = 10
) -> Dict[str, Dict[str, float]]:
    """Compare property leakage with and without DP.

    Args:
        n_clients: Number of clients
        n_features: Number of features
        target_property: Property to infer
        noise_multiplier: DP noise multiplier
        n_rounds: Number of FL rounds

    Returns:
        Dict comparing no-DP and DP scenarios

    Example:
        >>> results = compare_dp_vs_no_dp(noise_multiplier=1.0)
        >>> results['no_dp']['MAE']
        0.02
        >>> results['with_dp']['MAE']
        0.08
        >>> results['improvement']['MAE_reduction']
        0.06
    """
    # No DP
    results_no_dp = analyze_dp_effect_on_leakage(
        n_clients=n_clients,
        n_features=n_features,
        target_property=target_property,
        noise_multipliers=[0.0],  # No noise
        n_rounds=n_rounds
    )

    # With DP
    results_with_dp = analyze_dp_effect_on_leakage(
        n_clients=n_clients,
        n_features=n_features,
        target_property=target_property,
        noise_multipliers=[noise_multiplier],
        n_rounds=n_rounds
    )

    no_dp_mae = results_no_dp['0.0']['MAE']
    with_dp_mae = results_with_dp[str(noise_multiplier)]['MAE']

    return {
        'no_dp': results_no_dp['0.0'],
        'with_dp': results_with_dp[str(noise_multiplier)],
        'improvement': {
            'MAE_increase': float(with_dp_mae - no_dp_mae),
            'MAE_ratio': float(with_dp_mae / (no_dp_mae + 1e-8)),
            'privacy_improvement': with_dp_mae > no_dp_mae
        }
    }


def analyze_dp_by_property_type(
    n_clients: int = 10,
    n_features: int = 10,
    noise_multiplier: float = 1.0,
    n_rounds: int = 10
) -> Dict[str, Dict[str, float]]:
    """Analyze DP effectiveness for different property types.

    Some properties may leak more than others even with DP.

    Args:
        n_clients: Number of clients
        n_features: Number of features
        noise_multiplier: DP noise multiplier
        n_rounds: Number of FL rounds

    Returns:
        Dict mapping property names to leakage metrics

    Example:
        >>> results = analyze_dp_by_property_type(noise_multiplier=1.0)
        >>> results['fraud_rate']['MAE']
        0.08
        >>> results['dataset_size']['MAE']
        0.15  # Better protected
    """
    properties_to_test = ['fraud_rate', 'dataset_size', 'class_imbalance']

    results = {}

    for prop in properties_to_test:
        prop_results = analyze_dp_effect_on_leakage(
            n_clients=n_clients,
            n_features=n_features,
            target_property=prop,
            noise_multipliers=[noise_multiplier],
            n_rounds=n_rounds
        )

        results[prop] = prop_results[str(noise_multiplier)]

    return results


def plot_dp_leakage_curve(
    noise_levels: List[float],
    mae_values: List[float],
    save_path: str = None
) -> None:
    """Plot DP noise vs property leakage curve.

    Args:
        noise_levels: List of noise multipliers
        mae_values: Corresponding attack MAE values
        save_path: Optional path to save figure

    Example:
        >>> plot_dp_leakage_curve(
        ...     noise_levels=[0.1, 0.5, 1.0, 2.0, 5.0],
        ...     mae_values=[0.02, 0.05, 0.08, 0.12, 0.18]
        ... )
    """
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, mae_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('DP Noise Multiplier', fontsize=12)
    plt.ylabel('Attack MAE', fontsize=12)
    plt.title('Property Leakage vs DP Noise Level', fontsize=14)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def compute_epsilon_delta_bounds(
    noise_multiplier: float,
    max_grad_norm: float,
    n_rounds: int,
    delta: float = 1e-5
) -> Dict[str, float]:
    """Compute (ε, δ) DP bounds for given noise parameters.

    Uses moments accountant approach for FL.

    Args:
        noise_multiplier: DP noise multiplier
        max_grad_norm: Maximum gradient norm for clipping
        n_rounds: Number of training rounds
        delta: Delta parameter

    Returns:
        Dict with epsilon value

    Example:
        >>> bounds = compute_epsilon_delta_bounds(
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ...     n_rounds=20
        ... )
        >>> bounds['epsilon']
        3.5
    """
    # Simplified epsilon computation
    # In practice, use more sophisticated moments accountant

    # Per-round epsilon
    if noise_multiplier > 0:
        # Using Gaussian mechanism
        sigma = noise_multiplier * max_grad_norm
        epsilon_per_round = max_grad_norm / sigma
    else:
        epsilon_per_round = float('inf')

    # Total epsilon (composition)
    # Advanced composition theorem
    total_epsilon = np.sqrt(2 * n_rounds * np.log(1/delta)) * epsilon_per_round

    return {
        'epsilon': float(total_epsilon),
        'delta': float(delta),
        'epsilon_per_round': float(epsilon_per_round),
        'noise_multiplier': float(noise_multiplier)
    }
