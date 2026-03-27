"""
Client-Side Attack Scenario - Client observes only global model changes.

In this scenario, a malicious client attempts to infer properties of
OTHER clients' datasets by observing only how the global model changes
after each round of aggregation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from ..fl_system.server import FederatedServer
from ..fl_system.client import FederatedClient
from ..attacks.property_inference import PropertyInferenceAttack
from ..attacks.meta_classifier import PropertyMetaClassifier
from ..data_generation.synthetic_generator import generate_client_datasets
from ..metrics.attack_metrics import (
    compute_regression_metrics,
    compare_to_baseline
)


def setup_client_attack_scenario(
    n_clients: int = 10,
    n_features: int = 10,
    attacker_client_id: int = 0,
    property_variations: Dict[str, Any] = None,
    model_config: Dict[str, Any] = None,
    random_state: int = 42
) -> Tuple[FederatedServer, List[FederatedClient], FederatedClient]:
    """Setup federated learning system for client-side attack.

    Args:
        n_clients: Total number of clients
        n_features: Number of features
        attacker_client_id: ID of the malicious client
        property_variations: How properties vary across clients
        model_config: Model configuration
        random_state: Random seed

    Returns:
        (server, all_clients, attacker_client) tuple

    Example:
        >>> server, clients, attacker = setup_client_attack_scenario(
        ...     n_clients=10,
        ...     attacker_client_id=0
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

    # Generate client datasets
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

    # Create server
    server = FederatedServer(
        model_config=model_config,
        n_clients=n_clients,
        aggregation_method='fedavg'
    )

    return server, clients, clients[attacker_client_id]


def execute_client_attack(
    server: FederatedServer,
    clients: List[FederatedClient],
    attacker_client: FederatedClient,
    target_property: str,
    n_rounds: int = 10,
    train_test_split_round: int = 7,
    meta_classifier_type: str = 'rf_regressor'
) -> Dict[str, Any]:
    """Execute client-side property inference attack.

    The attacker observes global model changes and attempts to infer
    properties of other clients' datasets.

    Args:
        server: FL server instance
        clients: List of all clients
        attacker_client: The malicious client
        target_property: Property to infer
        n_rounds: Number of FL rounds
        train_test_split_round: Round to split train/test
        meta_classifier_type: Type of meta-classifier

    Returns:
        Dict with attack results

    Example:
        >>> results = execute_client_attack(
        ...     server, clients, attacker,
        ...     target_property='fraud_rate',
        ...     n_rounds=20
        ... )
        >>> results['test_metrics']['MAE']
        0.04
    """
    # Run FL and collect global model changes
    global_model_history = []
    client_properties_history = []

    for round_idx in range(n_rounds):
        # Run round
        metrics = server.run_round(clients)

        # Record global model
        global_weights = server.distribute_global_model()
        global_model_history.append(global_weights)

        # Record client properties (excluding attacker)
        round_props = []
        for client in clients:
            if client.client_id != attacker_client.client_id:
                props = client.get_dataset_properties()
                round_props.append(props[target_property])
        client_properties_history.append(round_props)

    # Compute global model changes
    global_changes = []
    for i in range(1, len(global_model_history)):
        change = global_model_history[i] - global_model_history[i - 1]
        global_changes.append(change)

    # Create training data
    # Each observation: (global_change, aggregated_property_of_others)
    train_updates = []
    train_props = []
    test_updates = []
    test_props = []

    for round_idx in range(len(global_changes)):
        if round_idx < train_test_split_round:
            train_updates.append(global_changes[round_idx])
            # Average property of other clients
            avg_prop = np.mean(client_properties_history[round_idx])
            train_props.append(avg_prop)
        else:
            test_updates.append(global_changes[round_idx])
            avg_prop = np.mean(client_properties_history[round_idx])
            test_props.append(avg_prop)

    train_updates = np.array(train_updates)
    train_props = np.array(train_props)
    test_updates = np.array(test_updates)
    test_props = np.array(test_props)

    # Train meta-classifier
    meta_classifier = PropertyMetaClassifier(
        property_name=target_property,
        model_type=meta_classifier_type
    )

    if len(train_updates) > 0:
        meta_classifier.train(train_updates, train_props)

    # Evaluate
    if len(test_updates) > 0:
        test_metrics = meta_classifier.evaluate(test_updates, test_props)
        baseline_comparison = compare_to_baseline(test_props, meta_classifier.predict(test_updates))
    else:
        test_metrics = {}
        baseline_comparison = {}

    return {
        'meta_classifier': meta_classifier,
        'train_metrics': meta_classifier.evaluate(train_updates, train_props) if len(train_updates) > 0 else {},
        'test_metrics': test_metrics,
        'baseline_comparison': baseline_comparison,
        'n_train_samples': len(train_updates),
        'n_test_samples': len(test_updates)
    }


def analyze_inference_per_target_client(
    server: FederatedServer,
    clients: List[FederatedClient],
    attacker_client: FederatedClient,
    target_property: str,
    n_rounds: int = 20
) -> Dict[int, Dict[str, float]]:
    """Analyze inference accuracy for each target client individually.

    Args:
        server: FL server instance
        clients: List of all clients
        attacker_client: The malicious client
        target_property: Property to infer
        n_rounds: Number of FL rounds

    Returns:
        Dict mapping target client IDs to error metrics

    Example:
        >>> results = analyze_inference_per_target_client(
        ...     server, clients, attacker, 'fraud_rate', n_rounds=20
        ... )
        >>> results[1]['MAE']
        0.05
        >>> results[5]['MAE']
        0.08
    """
    # Run FL
    global_model_history = []

    for round_idx in range(n_rounds):
        server.run_round(clients)
        global_weights = server.distribute_global_model()
        global_model_history.append(global_weights)

    # Compute global model changes
    global_changes = []
    for i in range(1, len(global_model_history)):
        change = global_model_history[i] - global_model_history[i - 1]
        global_changes.append(change)

    # Analyze each target client
    client_errors = {}

    for target_client in clients:
        if target_client.client_id == attacker_client.client_id:
            continue

        # Get true property value
        true_value = target_client.get_dataset_properties()[target_property]

        # Create observations where this client participated
        target_updates = []
        for round_idx in range(len(global_changes)):
            target_updates.append(global_changes[round_idx])

        # Simple prediction: use mean of observed changes as proxy
        # (In practice, would train separate meta-classifier per target)
        if len(target_updates) > 0:
            target_updates = np.array(target_updates)
            # Use change magnitude as simple predictor
            change_magnitudes = np.linalg.norm(target_updates, axis=1)

            # Normalize to property range
            pred_mean = change_magnitudes.mean()
            pred_normalized = pred_mean / (change_magnitudes.max() + 1e-8)

            # Map to property range
            prop_min, prop_max = 0.01, 0.2  # Assume fraud rate range
            predicted = prop_min + pred_normalized * (prop_max - prop_min)

            client_errors[target_client.client_id] = {
                'MAE': float(np.abs(predicted - true_value)),
                'true_value': float(true_value),
                'predicted': float(predicted)
            }

    return client_errors


def evaluate_collusion_attack(
    server: FederatedServer,
    clients: List[FederatedClient],
    attacker_client_ids: List[int],
    target_property: str,
    n_rounds: int = 10
) -> Dict[str, Any]:
    """Evaluate attack where multiple clients collude.

    Colluding clients share their observed global model changes to
    improve inference accuracy.

    Args:
        server: FL server instance
        clients: List of all clients
        attacker_client_ids: IDs of colluding clients
        target_property: Property to infer
        n_rounds: Number of FL rounds

    Returns:
        Dict with collusion attack results

    Example:
        >>> results = evaluate_collusion_attack(
        ...     server, clients,
        ...     attacker_client_ids=[0, 1, 2],
        ...     target_property='fraud_rate'
        ... )
        >>> results['MAE']
        0.03
    """
    # Run FL
    global_model_history = []

    for round_idx in range(n_rounds):
        server.run_round(clients)
        global_weights = server.distribute_global_model()
        global_model_history.append(global_weights)

    # Compute global changes
    global_changes = []
    for i in range(1, len(global_model_history)):
        change = global_model_history[i] - global_model_history[i - 1]
        global_changes.append(change)

    # Aggregate observations from colluding clients
    # (In practice, each client sees same global changes, so collusion
    #  doesn't help for global model observations)
    # However, colluding clients could share local training losses, etc.

    # Get properties of non-colluding clients
    target_properties = []
    for client in clients:
        if client.client_id not in attacker_client_ids:
            props = client.get_dataset_properties()
            target_properties.append(props[target_property])

    # Simple prediction based on global changes
    if len(global_changes) > 0 and len(target_properties) > 0:
        change_magnitudes = [np.linalg.norm(c) for c in global_changes]
        avg_change = np.mean(change_magnitudes)

        # Normalize to property range
        prop_min, prop_max = 0.01, 0.2
        pred_normalized = avg_change / (max(change_magnitudes) + 1e-8)
        predicted = prop_min + pred_normalized * (prop_max - prop_min)

        true_mean = np.mean(target_properties)

        return {
            'MAE': float(np.abs(predicted - true_mean)),
            'true_mean': float(true_mean),
            'predicted': float(predicted),
            'n_colluders': len(attacker_client_ids),
            'n_targets': len(target_properties)
        }
    else:
        return {
            'MAE': None,
            'true_mean': None,
            'predicted': None,
            'n_colluders': len(attacker_client_ids),
            'n_targets': 0
        }


def compare_server_vs_client_attack(
    n_clients: int = 10,
    n_features: int = 10,
    target_property: str = 'fraud_rate',
    n_rounds: int = 20
) -> Dict[str, Dict[str, Any]]:
    """Compare server-side and client-side attack effectiveness.

    Args:
        n_clients: Number of clients
        n_features: Number of features
        target_property: Property to infer
        n_rounds: Number of FL rounds

    Returns:
        Dict comparing server and client attacks

    Example:
        >>> results = compare_server_vs_client_attack(
        ...     n_clients=10, target_property='fraud_rate'
        ... )
        >>> results['server_attack']['MAE']
        0.02
        >>> results['client_attack']['MAE']
        0.05
    """
    from .server_attack import setup_server_attack_scenario, execute_server_attack

    # Setup
    property_variations = {
        'fraud_rate': {'type': 'uniform', 'min': 0.01, 'max': 0.2},
        'dataset_size': {'type': 'normal', 'mean': 1000, 'std': 300}
    }

    model_config = {
        'type': 'logistic_regression',
        'input_dim': n_features
    }

    # Server attack
    server, clients = setup_server_attack_scenario(
        n_clients=n_clients,
        n_features=n_features,
        property_variations=property_variations,
        model_config=model_config
    )

    server_results = execute_server_attack(
        server=server,
        clients=clients,
        target_property=target_property,
        n_rounds=n_rounds
    )

    # Client attack
    _, clients2, attacker = setup_client_attack_scenario(
        n_clients=n_clients,
        n_features=n_features,
        attacker_client_id=0,
        property_variations=property_variations,
        model_config=model_config
    )

    client_results = execute_client_attack(
        server=server,
        clients=clients2,
        attacker_client=attacker,
        target_property=target_property,
        n_rounds=n_rounds
    )

    return {
        'server_attack': {
            'MAE': server_results['test_metrics'].get('MAE', None),
            'R2': server_results['test_metrics'].get('R2', None),
            'attack_better': server_results['baseline_comparison'].get('attack_better', None)
        },
        'client_attack': {
            'MAE': client_results['test_metrics'].get('MAE', None),
            'R2': client_results['test_metrics'].get('R2', None),
            'attack_better': client_results['baseline_comparison'].get('attack_better', None)
        }
    }
