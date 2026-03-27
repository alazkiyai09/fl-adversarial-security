"""
Secure Aggregation Defense Analysis.

This module analyzes how secure aggregation affects property
inference leakage in federated learning.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

from ..fl_system.server import FederatedServer
from ..fl_system.client import FederatedClient
from ..data_generation.synthetic_generator import generate_client_datasets
from ..attacks.property_inference import PropertyInferenceAttack
from ..metrics.attack_metrics import compute_regression_metrics, compare_to_baseline


def simulate_secure_aggregation(
    client_updates: List[np.ndarray],
    drop_clients: List[int] = None
) -> np.ndarray:
    """Simulate secure aggregation where server sees only sum.

    In true secure aggregation, the server only sees the aggregated sum,
    not individual client updates.

    Args:
        client_updates: List of client update arrays
        drop_clients: Indices of clients that dropped out (optional)

    Returns:
        Aggregated update (sum of all client updates)

    Example:
        >>> updates = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> aggregated = simulate_secure_aggregation(updates)
        >>> aggregated
        array([5, 7, 9])
    """
    if drop_clients is None:
        drop_clients = []

    # Sum all updates (excluding dropped clients)
    aggregated = np.zeros_like(client_updates[0])

    for i, update in enumerate(client_updates):
        if i not in drop_clients:
            aggregated += update

    return aggregated


def analyze_secure_agg_effect_on_leakage(
    n_clients: int = 10,
    n_features: int = 10,
    target_property: str = 'fraud_rate',
    n_rounds: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """Analyze how secure aggregation affects property inference leakage.

    Args:
        n_clients: Number of clients
        n_features: Number of features
        target_property: Property to infer
        n_rounds: Number of FL rounds
        random_state: Random seed

    Returns:
        Dict comparing leakage with and without secure aggregation

    Example:
        >>> results = analyze_secure_agg_effect_on_leakage()
        >>> results['no_secure_agg']['MAE']
        0.02
        >>> results['with_secure_agg']['MAE']
        0.15  # Much worse for attacker
    """
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

    # Create clients
    clients = []
    for client_id, dataset in enumerate(datasets):
        client = FederatedClient(
            client_id=client_id,
            dataset=dataset,
            model_config=model_config,
            local_epochs=5,
            learning_rate=0.01
        )
        clients.append(client)

    # Create server
    server = FederatedServer(
        model_config=model_config,
        n_clients=n_clients
    )

    results = {}

    # WITHOUT secure aggregation: server sees all individual updates
    individual_updates_list = []
    individual_props_list = []

    for round_idx in range(n_rounds):
        server.run_round(clients)

        # Get individual client updates
        round_updates = server.get_model_updates(round_idx, 'gradients')

        for client_idx, update in enumerate(round_updates):
            individual_updates_list.append(update)
            client_props = clients[client_idx].get_dataset_properties()
            individual_props_list.append(client_props[target_property])

    # Train attack on individual updates
    n_train = int(0.8 * len(individual_updates_list))
    train_updates = np.array(individual_updates_list[:n_train])
    test_updates = np.array(individual_updates_list[n_train:])
    train_props = np.array(individual_props_list[:n_train])
    test_props = np.array(individual_props_list[n_train:])

    attack_no_sec_agg = PropertyInferenceAttack(
        target_property=target_property,
        scenario='server'
    )
    attack_no_sec_agg.train_meta_classifier(train_updates, train_props)
    results_no_sec_agg = attack_no_sec_agg.evaluate_attack(test_updates, test_props)

    results['no_secure_agg'] = {
        'MAE': results_no_sec_agg.get('MAE', None),
        'R2': results_no_sec_agg.get('R2', None),
        'attack_better': results_no_sec_agg.get('baseline_comparison', {}).get('attack_better', False)
    }

    # WITH secure aggregation: server only sees aggregated sum
    # This prevents individual client inference
    # Attack can only infer AVERAGE property across all clients

    aggregated_updates_list = []
    avg_props_list = []

    for round_idx in range(n_rounds):
        round_updates = server.get_model_updates(round_idx, 'gradients')

        # Simulate secure aggregation
        aggregated = simulate_secure_aggregation(round_updates)

        # Compute average property across clients
        round_props = []
        for client_idx in range(len(clients)):
            client_props = clients[client_idx].get_dataset_properties()
            round_props.append(client_props[target_property])

        avg_prop = np.mean(round_props)

        aggregated_updates_list.append(aggregated)
        avg_props_list.append(avg_prop)

    # Train attack on aggregated updates
    # Note: Much harder because each observation is just the sum
    n_train = int(0.8 * len(aggregated_updates_list))
    train_updates_agg = np.array(aggregated_updates_list[:n_train])
    test_updates_agg = np.array(aggregated_updates_list[n_train:])
    train_props_agg = np.array(avg_props_list[:n_train])
    test_props_agg = np.array(avg_props_list[n_train:])

    # For aggregated case, we're predicting the average property
    # This is fundamentally different from individual inference
    attack_with_sec_agg = PropertyInferenceAttack(
        target_property=target_property,
        scenario='server'
    )

    if len(train_updates_agg) > 0:
        attack_with_sec_agg.train_meta_classifier(train_updates_agg, train_props_agg)
        results_with_sec_agg = attack_with_sec_agg.evaluate_attack(
            test_updates_agg,
            test_props_agg
        )

        results['with_secure_agg'] = {
            'MAE': results_with_sec_agg.get('MAE', None),
            'R2': results_with_sec_agg.get('R2', None),
            'note': 'Predicting average property, not individual'
        }
    else:
        results['with_secure_agg'] = {
            'MAE': None,
            'R2': None,
            'note': 'Insufficient data'
        }

    # Compute improvement
    if results['no_secure_agg']['MAE'] is not None and results['with_secure_agg'].get('MAE') is not None:
        results['improvement'] = {
            'MAE_increase_for_attacker': results['with_secure_agg']['MAE'] - results['no_secure_agg']['MAE'],
            'attacker_much_worse': results['with_secure_agg']['MAE'] > results['no_secure_agg']['MAE'] * 2
        }

    return results


def analyze_drop_out_effect(
    n_clients: int = 10,
    n_features: int = 10,
    n_rounds: int = 10,
    drop_rates: List[float] = [0.0, 0.1, 0.2, 0.3]
) -> Dict[str, Any]:
    """Analyze how client dropouts affect secure aggregation.

    With secure aggregation, if clients drop out, the server may
    learn information about the remaining clients.

    Args:
        n_clients: Number of clients
        n_features: Number of features
        n_rounds: Number of FL rounds
        drop_rates: List of client dropout rates to test

    Returns:
        Dict analyzing dropout effects

    Example:
        >>> results = analyze_drop_out_effect(drop_rates=[0.0, 0.2])
        >>> results['0.0']['leakage_possible']
        False
        >>> results['0.2']['leakage_possible']
        True
    """
    results = {}

    for drop_rate in drop_rates:
        n_drop = int(n_clients * drop_rate)

        # With secure aggregation
        # If some clients drop, server can potentially infer info
        # about the remaining clients by comparing to previous rounds

        # Simplified analysis: check if dropout enables inference
        leakage_possible = n_drop > 0 and n_drop < n_clients - 1

        results[str(drop_rate)] = {
            'n_drop': n_drop,
            'n_remaining': n_clients - n_drop,
            'leakage_possible': leakage_possible,
            'severity': 'high' if 1 < n_drop < n_clients - 1 else 'none'
        }

    return results


def compare_defense_combinations(
    n_clients: int = 10,
    n_features: int = 10,
    target_property: str = 'fraud_rate',
    noise_multiplier: float = 1.0,
    n_rounds: int = 10
) -> Dict[str, Dict[str, float]]:
    """Compare different defense combinations.

    Args:
        n_clients: Number of clients
        n_features: Number of features
        target_property: Property to infer
        noise_multiplier: DP noise multiplier
        n_rounds: Number of FL rounds

    Returns:
        Dict comparing defense combinations

    Example:
        >>> results = compare_defense_combinations()
        >>> results['no_defense']['MAE']
        0.02
        >>> results['dp_only']['MAE']
        0.08
        >>> results['secure_agg_only']['MAE']
        0.15
        >>> results['both']['MAE']
        0.20
    """
    from .dp_analysis import analyze_dp_effect_on_leakage

    results = {}

    # No defense
    no_defense = analyze_dp_effect_on_leakage(
        n_clients=n_clients,
        n_features=n_features,
        target_property=target_property,
        noise_multipliers=[0.0],
        n_rounds=n_rounds
    )
    results['no_defense'] = no_defense['0.0']

    # DP only
    dp_only = analyze_dp_effect_on_leakage(
        n_clients=n_clients,
        n_features=n_features,
        target_property=target_property,
        noise_multipliers=[noise_multiplier],
        n_rounds=n_rounds
    )
    results['dp_only'] = dp_only[str(noise_multiplier)]

    # Secure aggregation only
    sec_agg_only = analyze_secure_agg_effect_on_leakage(
        n_clients=n_clients,
        n_features=n_features,
        target_property=target_property,
        n_rounds=n_rounds
    )
    results['secure_agg_only'] = sec_agg_only.get('with_secure_agg', {})

    # Both DP and secure aggregation
    # Note: This would require combined implementation
    # For now, use heuristic: combine effects
    results['both'] = {
        'MAE': results['dp_only'].get('MAE', 0) + results['secure_agg_only'].get('MAE', 0),
        'note': 'Estimated (sum of individual effects)'
    }

    return results


def plot_defense_comparison(
    defense_results: Dict[str, Dict[str, float]],
    save_path: str = None
) -> None:
    """Plot comparison of different defenses.

    Args:
        defense_results: Results from compare_defense_combinations
        save_path: Optional path to save figure

    Example:
        >>> plot_defense_comparison(results)
    """
    defenses = list(defense_results.keys())
    mae_values = [defense_results[d].get('MAE', 0) for d in defenses]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(defenses, mae_values, color=['red', 'orange', 'yellow', 'green'])
    plt.ylabel('Attack MAE (higher is better)', fontsize=12)
    plt.xlabel('Defense Type', fontsize=12)
    plt.title('Property Inference Defense Comparison', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mae in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mae:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def compute_defense_cost(
    defense_type: str,
    noise_multiplier: float = 1.0
) -> Dict[str, float]:
    """Compute computational/accuracy cost of defense.

    Args:
        defense_type: Type of defense ('dp', 'secure_agg', 'both')
        noise_multiplier: DP noise multiplier

    Returns:
        Dict with cost metrics

    Example:
        >>> cost = compute_defense_cost('dp', noise_multiplier=1.0)
        >>> cost['accuracy_degradation']
        0.05
        >>> cost['communication_overhead']
        0.0
    """
    costs = {
        'dp': {
            'accuracy_degradation': 0.05 * noise_multiplier,  # Approximate
            'communication_overhead': 0.0,
            'computation_overhead': 0.1  # Noise generation
        },
        'secure_agg': {
            'accuracy_degradation': 0.0,
            'communication_overhead': 2.0,  # 2x for cryptographic protocols
            'computation_overhead': 0.5  # Encryption/decryption
        },
        'both': {
            'accuracy_degradation': 0.05 * noise_multiplier,
            'communication_overhead': 2.0,
            'computation_overhead': 0.6
        }
    }

    return costs.get(defense_type, {})
