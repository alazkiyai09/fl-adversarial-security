"""
Server-Side Attack Scenario - Server observes all client updates.

In this scenario, the malicious (honest-but-curious) server receives
individual model updates from all clients and attempts to infer their
dataset properties.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from ..fl_system.server import FederatedServer, MaliciousServer
from ..fl_system.client import FederatedClient
from ..attacks.property_inference import PropertyInferenceAttack
from ..attacks.meta_classifier import PropertyMetaClassifier
from ..data_generation.synthetic_generator import generate_client_datasets
from ..metrics.attack_metrics import (
    compute_regression_metrics,
    compare_to_baseline,
    compute_temporal_metrics
)


def setup_server_attack_scenario(
    n_clients: int = 10,
    n_features: int = 10,
    property_variations: Dict[str, Any] = None,
    model_config: Dict[str, Any] = None,
    random_state: int = 42
) -> Tuple[MaliciousServer, List[FederatedClient]]:
    """Setup federated learning system for server-side attack.

    Args:
        n_clients: Number of clients (banks)
        n_features: Number of features
        property_variations: How properties vary across clients
        model_config: Model configuration
        random_state: Random seed

    Returns:
        (malicious_server, clients) tuple

    Example:
        >>> server, clients = setup_server_attack_scenario(
        ...     n_clients=5,
        ...     property_variations={
        ...         'fraud_rate': {'type': 'uniform', 'min': 0.01, 'max': 0.2}
        ...     }
        ... )
    """
    if property_variations is None:
        property_variations = {
            'fraud_rate': {'type': 'uniform', 'min': 0.01, 'max': 0.2},
            'dataset_size': {'type': 'normal', 'mean': 1000, 'std': 300}
        }

    if model_config is None:
        model_config = {
            'type': 'logistic_regression',
            'input_dim': n_features
        }

    # Generate client datasets with varied properties
    datasets, properties = generate_client_datasets(
        n_clients=n_clients,
        base_n_samples=1000,
        n_features=n_features,
        property_variations=property_variations,
        random_state=random_state
    )

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

    # Create malicious server
    server = MaliciousServer(
        model_config=model_config,
        n_clients=n_clients,
        aggregation_method='fedavg'
    )

    return server, clients


def execute_server_attack(
    server: MaliciousServer,
    clients: List[FederatedClient],
    target_property: str,
    n_rounds: int = 10,
    train_test_split_round: int = 7,
    meta_classifier_type: str = 'rf_regressor'
) -> Dict[str, Any]:
    """Execute server-side property inference attack.

    Args:
        server: Malicious server instance
        clients: List of clients
        target_property: Property to infer (e.g., 'fraud_rate')
        n_rounds: Number of FL rounds to run
        train_test_split_round: Round number to split train/test data
        meta_classifier_type: Type of meta-classifier

    Returns:
        Dict with attack results

    Example:
        >>> results = execute_server_attack(
        ...     server, clients,
        ...     target_property='fraud_rate',
        ...     n_rounds=20
        ... )
        >>> results['test_metrics']['MAE']
        0.02
    """
    # Run FL and collect observations
    for round_idx in range(n_rounds):
        server.run_round_with_observation(clients)

    # Extract attack dataset
    train_updates, train_props = server.extract_attack_dataset(
        target_property=target_property,
        rounds=list(range(train_test_split_round))
    )

    test_updates, test_props = server.extract_attack_dataset(
        target_property=target_property,
        rounds=list(range(train_test_split_round, n_rounds))
    )

    # Train meta-classifier
    meta_classifier = PropertyMetaClassifier(
        property_name=target_property,
        model_type=meta_classifier_type
    )

    meta_classifier.train(train_updates, train_props)

    # Evaluate
    test_metrics = meta_classifier.evaluate(test_updates, test_props)
    baseline_comparison = compare_to_baseline(test_props, meta_classifier.predict(test_updates))

    return {
        'meta_classifier': meta_classifier,
        'train_metrics': meta_classifier.evaluate(train_updates, train_props),
        'test_metrics': test_metrics,
        'baseline_comparison': baseline_comparison,
        'n_train_samples': len(train_updates),
        'n_test_samples': len(test_updates)
    }


def analyze_temporal_leakage(
    server: MaliciousServer,
    clients: List[FederatedClient],
    target_property: str,
    n_rounds: int = 20,
    meta_classifier_type: str = 'rf_regressor'
) -> Dict[str, Any]:
    """Analyze how property leakage changes over FL rounds.

    Args:
        server: Malicious server instance
        clients: List of clients
        target_property: Property to infer
        n_rounds: Number of FL rounds
        meta_classifier_type: Type of meta-classifier

    Returns:
        Dict with temporal analysis results

    Example:
        >>> results = analyze_temporal_leakage(
        ...     server, clients, 'fraud_rate', n_rounds=20
        ... )
        >>> results['MAE_by_round'][0]  # Early rounds
        0.05
        >>> results['MAE_by_round'][-1]  # Late rounds
        0.02
    """
    # Run FL and collect observations
    for round_idx in range(n_rounds):
        server.run_round_with_observation(clients)

    # Extract data for each round
    round_predictions = []
    round_true_values = []

    for round_idx in range(n_rounds):
        updates, props = server.extract_attack_dataset(
            target_property=target_property,
            rounds=[round_idx]
        )

        if len(updates) > 0:
            # Train on data up to this round
            train_updates, train_props = server.extract_attack_dataset(
                target_property=target_property,
                rounds=list(range(round_idx))
            )

            if len(train_updates) > 0:
                meta_classifier = PropertyMetaClassifier(
                    property_name=target_property,
                    model_type=meta_classifier_type
                )
                meta_classifier.train(train_updates, train_props)

                # Predict on current round
                predictions = meta_classifier.predict(updates)
                round_predictions.append(predictions)
                round_true_values.append(props)

    # Compute temporal metrics
    temporal_metrics = compute_temporal_metrics(round_true_values, round_predictions)

    return {
        'temporal_metrics': temporal_metrics,
        'MAE_by_round': temporal_metrics['MAE_by_round'],
        'R2_by_round': temporal_metrics['R2_by_round'],
        'improvement_early_vs_late': temporal_metrics['MAE_improvement_early_vs_late']
    }


def compare_client_selection_strategies(
    server: MaliciousServer,
    clients: List[FederatedClient],
    target_property: str,
    n_rounds: int = 20,
    client_fractions: List[float] = [0.3, 0.5, 0.7, 1.0]
) -> Dict[str, Any]:
    """Compare attack performance with different client selection strategies.

    Args:
        server: Malicious server instance
        clients: List of clients
        target_property: Property to infer
        n_rounds: Number of FL rounds
        client_fractions: List of client fractions to test

    Returns:
        Dict comparing strategies

    Example:
        >>> results = compare_client_selection_strategies(
        ...     server, clients, 'fraud_rate',
        ...     client_fractions=[0.5, 1.0]
        ... )
        >>> results['0.5']['MAE']
        0.04
        >>> results['1.0']['MAE']
        0.02
    """
    results = {}

    for fraction in client_fractions:
        # Reset server
        from ..fl_system.server import MaliciousServer
        model_config = server.model_config
        test_server = MaliciousServer(
            model_config=model_config,
            n_clients=len(clients)
        )

        # Run attack
        attack_results = execute_server_attack(
            server=test_server,
            clients=clients,
            target_property=target_property,
            n_rounds=n_rounds
        )

        results[str(fraction)] = {
            'MAE': attack_results['test_metrics']['MAE'],
            'R2': attack_results['test_metrics']['R2'],
            'baseline_better': not attack_results['baseline_comparison']['attack_better']
        }

    return results


def evaluate_per_client_accuracy(
    server: MaliciousServer,
    clients: List[FederatedClient],
    target_property: str,
    n_rounds: int = 10
) -> Dict[int, Dict[str, float]]:
    """Evaluate attack accuracy for each individual client.

    Args:
        server: Malicious server instance
        clients: List of clients
        target_property: Property to infer
        n_rounds: Number of FL rounds

    Returns:
        Dict mapping client IDs to error metrics

    Example:
        >>> results = evaluate_per_client_accuracy(
        ...     server, clients, 'fraud_rate', n_rounds=10
        ... )
        >>> results[0]['MAE']
        0.01
        >>> results[4]['MAE']
        0.03
    """
    # Run FL
    for round_idx in range(n_rounds):
        server.run_round_with_observation(clients)

    # Train meta-classifier on all data
    train_updates, train_props = server.extract_attack_dataset(
        target_property=target_property
    )

    meta_classifier = PropertyMetaClassifier(
        property_name=target_property,
        model_type='rf_regressor'
    )
    meta_classifier.train(train_updates, train_props)

    # Evaluate per client
    client_errors = {}

    for client in clients:
        client_props = client.get_dataset_properties()
        true_value = client_props[target_property]

        # Get predictions for this client across rounds
        client_updates = []
        for round_idx in range(n_rounds):
            round_updates, _ = server.extract_attack_dataset(
                target_property=target_property,
                rounds=[round_idx]
            )

            if client.client_id < len(round_updates):
                client_updates.append(round_updates[client.client_id])

        if len(client_updates) > 0:
            client_updates = np.array(client_updates)
            predictions = meta_classifier.predict(client_updates)

            # Compute errors
            errors = np.abs(predictions - true_value)
            client_errors[client.client_id] = {
                'MAE': float(errors.mean()),
                'std': float(errors.std()),
                'true_value': float(true_value),
                'predicted_mean': float(predictions.mean()),
                'n_observations': len(predictions)
            }

    return client_errors
