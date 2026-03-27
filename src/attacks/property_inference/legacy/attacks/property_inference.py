"""
Property Inference Attack - Main attack orchestrator.

This module implements the complete property inference attack pipeline,
from training data generation to attack execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
import yaml

from .meta_classifier import PropertyMetaClassifier, MultiOutputMetaClassifier
from .property_extractor import extract_all_properties, get_property_value
from ..fl_system.server import FederatedServer, MaliciousServer
from ..fl_system.client import FederatedClient
from ..data_generation.synthetic_generator import generate_fraud_dataset
from ..data_generation.property_varier import PropertyVarier
from ..metrics.attack_metrics import (
    compute_regression_metrics,
    compare_to_baseline,
    compute_rank_correlation,
    bootstrap_confidence_interval
)


class PropertyInferenceAttack:
    """Property inference attack on federated learning.

    This attack learns to infer dataset properties (fraud rate, data volume,
    feature distributions) from observed model updates.

    Example:
        >>> attack = PropertyInferenceAttack(
        ...     target_property='fraud_rate',
        ...     scenario='server'
        ... )
        >>> attack.train_meta_classifier(n_datasets=500)
        >>> predicted = attack.execute_attack(observed_updates)
    """

    def __init__(
        self,
        target_property: str,
        scenario: str = 'server',
        meta_classifier_type: str = 'rf_regressor',
        config_path: Optional[str] = None
    ):
        """Initialize property inference attack.

        Args:
            target_property: Property to infer ('fraud_rate', 'dataset_size', etc.)
            scenario: Attack scenario
                - 'server': Server observes all individual client updates
                - 'client': Client observes only global model changes
            meta_classifier_type: Type of meta-classifier
            config_path: Path to configuration file (optional)
        """
        self.target_property = target_property
        self.scenario = scenario
        self.meta_classifier_type = meta_classifier_type

        # Load config if provided
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        # Initialize meta-classifier
        self.meta_classifier = PropertyMetaClassifier(
            property_name=target_property,
            model_type=meta_classifier_type,
            model_params=self.config.get('meta_classifier', {}).get(meta_classifier_type, {})
        )

        # Attack state
        self.is_trained = False
        self.attack_data = {
            'updates': None,
            'properties': None
        }

    def generate_attack_data(
        self,
        fl_simulator: Callable,
        n_datasets: int,
        n_rounds: int = 1,
        property_ranges: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for meta-classifier.

        Runs FL simulations with varied datasets, records (update, property) pairs.

        Args:
            fl_simulator: Function that runs FL simulation
                Signature: (dataset_properties) -> (server, clients)
            n_datasets: Number of synthetic datasets to generate
            n_rounds: Number of FL rounds per dataset
            property_ranges: Ranges for property variation

        Returns:
            (updates, properties) tuple

        Example:
            >>> def simulate(prop):
            ...     # Run FL with specified properties
            ...     return server, clients
            >>> updates, props = attack.generate_attack_data(
            ...     fl_simulator=simulate,
            ...     n_datasets=100
            ... )
        """
        all_updates = []
        all_properties = []

        # Create property varier
        if property_ranges is None:
            property_ranges = {
                'fraud_rate': {'min': 0.01, 'max': 0.2},
                'dataset_size': {'min': 100, 'max': 5000}
            }

        varier = PropertyVarier(
            n_features=self.config.get('n_features', 10),
            base_config={'random_state': 42}
        )

        # Generate datasets with varied properties
        datasets, properties_list = varier.vary_multiple_properties(
            property_ranges=property_ranges,
            n_datasets=n_datasets,
            random_state=42
        )

        # Run FL simulation for each dataset
        for i, (dataset, props) in enumerate(zip(datasets, properties_list)):
            # Run FL
            server, clients = fl_simulator(dataset)

            # Extract updates based on scenario
            if self.scenario == 'server':
                # Server observes all client updates
                for round_idx in range(n_rounds):
                    client_updates = server.get_model_updates(
                        round_idx,
                        update_type='gradients'
                    )
                    for update in client_updates:
                        all_updates.append(update)
                        all_properties.append(props[self.target_property])

            elif self.scenario == 'client':
                # Client observes only global model changes
                for round_idx in range(n_rounds):
                    global_weights = server.distribute_global_model()
                    # Client sees the difference between rounds
                    if round_idx > 0:
                        prev_weights = server.history['client_updates'][round_idx - 1][0]['weights']
                        update = global_weights - prev_weights
                        all_updates.append(update)
                        all_properties.append(props[self.target_property])

        self.attack_data['updates'] = np.array(all_updates)
        self.attack_data['properties'] = np.array(all_properties)

        return self.attack_data['updates'], self.attack_data['properties']

    def train_meta_classifier(
        self,
        updates: np.ndarray,
        properties: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """Train meta-classifier.

        Args:
            updates: Model updates from FL
            properties: True property values
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        metrics = self.meta_classifier.train(updates, properties, validation_split)
        self.is_trained = True
        return metrics

    def execute_attack(
        self,
        observed_updates: np.ndarray
    ) -> np.ndarray:
        """Predict properties from observed updates.

        Args:
            observed_updates: Model updates from target FL system

        Returns:
            Predicted property values

        Example:
            >>> observed = np.random.randn(10, 100)
            >>> predictions = attack.execute_attack(observed)
            >>> predictions.shape
            (10,)
        """
        if not self.is_trained:
            raise RuntimeError("Meta-classifier must be trained before attack execution")

        return self.meta_classifier.predict(observed_updates)

    def evaluate_attack(
        self,
        test_updates: np.ndarray,
        test_properties: np.ndarray,
        compute_baseline: bool = True
    ) -> Dict[str, Any]:
        """Evaluate attack performance on test data.

        Args:
            test_updates: Test model updates
            test_properties: True property values
            compute_baseline: Whether to compare to baseline

        Returns:
            Dict with evaluation metrics

        Example:
            >>> results = attack.evaluate_attack(test_updates, test_properties)
            >>> results['MAE']
            0.02
            >>> results['baseline_comparison']['attack_better']
            True
        """
        # Predict
        predictions = self.execute_attack(test_updates)

        # Compute metrics
        metrics = compute_regression_metrics(test_properties, predictions)
        metrics['rank_correlation'] = compute_rank_correlation(test_properties, predictions)

        # Compute confidence intervals
        mae_mean, mae_lower, mae_upper = bootstrap_confidence_interval(
            test_properties,
            predictions,
            lambda y_true, y_pred: np.abs(y_true - y_pred).mean(),
            n_bootstrap=1000
        )
        metrics['MAE_CI'] = (mae_lower, mae_upper)

        # Compare to baseline
        if compute_baseline:
            baseline_comparison = compare_to_baseline(test_properties, predictions)
            metrics['baseline_comparison'] = baseline_comparison

        return metrics

    def save_meta_classifier(self, filepath: str) -> None:
        """Save trained meta-classifier.

        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise RuntimeError("Meta-classifier must be trained before saving")

        self.meta_classifier.save(filepath)

    def load_meta_classifier(self, filepath: str) -> None:
        """Load trained meta-classifier.

        Args:
            filepath: Path to saved model
        """
        self.meta_classifier = PropertyMetaClassifier.load(filepath)
        self.is_trained = True


class FraudRateInferenceAttack(PropertyInferenceAttack):
    """Specialized attack for inferring fraud rates."""

    def __init__(
        self,
        scenario: str = 'server',
        meta_classifier_type: str = 'rf_regressor',
        config_path: Optional[str] = None
    ):
        """Initialize fraud rate inference attack."""
        super().__init__(
            target_property='fraud_rate',
            scenario=scenario,
            meta_classifier_type=meta_classifier_type,
            config_path=config_path
        )


class DataVolumeInferenceAttack(PropertyInferenceAttack):
    """Specialized attack for inferring dataset sizes."""

    def __init__(
        self,
        scenario: str = 'server',
        meta_classifier_type: str = 'rf_regressor',
        config_path: Optional[str] = None
    ):
        """Initialize data volume inference attack."""
        super().__init__(
            target_property='dataset_size',
            scenario=scenario,
            meta_classifier_type=meta_classifier_type,
            config_path=config_path
        )


def execute_attack_on_fl_system(
    server: FederatedServer,
    clients: List[FederatedClient],
    target_property: str,
    attack_rounds: Optional[List[int]] = None,
    meta_classifier_type: str = 'rf_regressor'
) -> Tuple[PropertyInferenceAttack, Dict[str, Any]]:
    """Execute property inference attack on an existing FL system.

    This convenience function extracts attack data from a running FL system
    and evaluates the attack.

    Args:
        server: FL server instance
        clients: List of FL client instances
        target_property: Property to infer
        attack_rounds: Rounds to analyze (None for all)
        meta_classifier_type: Type of meta-classifier

    Returns:
        (attack_instance, evaluation_results) tuple

    Example:
        >>> attack, results = execute_attack_on_fl_system(
        ...     server=server,
        ...     clients=clients,
        ...     target_property='fraud_rate'
        ... )
        >>> results['MAE']
        0.03
    """
    # Extract updates and properties from FL history
    if attack_rounds is None:
        attack_rounds = range(len(server.history['client_updates']))

    updates_list = []
    properties_list = []

    for round_idx in attack_rounds:
        client_updates = server.history['client_updates'][round_idx]

        for client_update in client_updates:
            client_id = client_update['client_id']
            update = client_update['update']

            # Get client properties
            client_props = clients[client_id].get_dataset_properties()
            prop_value = get_property_value(client_props, target_property)

            updates_list.append(update)
            properties_list.append(prop_value)

    updates = np.array(updates_list)
    properties = np.array(properties_list)

    # Split into train/test
    n_train = int(0.8 * len(updates))
    train_updates, test_updates = updates[:n_train], updates[n_train:]
    train_props, test_props = properties[:n_train], properties[n_train:]

    # Create and train attack
    attack = PropertyInferenceAttack(
        target_property=target_property,
        scenario='server',
        meta_classifier_type=meta_classifier_type
    )

    attack.train_meta_classifier(train_updates, train_props)

    # Evaluate
    results = attack.evaluate_attack(test_updates, test_props)

    return attack, results


def analyze_property_leakage(
    server: FederatedServer,
    clients: List[FederatedClient],
    properties: List[str],
    meta_classifier_type: str = 'rf_regressor'
) -> Dict[str, Dict[str, Any]]:
    """Analyze leakage for multiple properties.

    Args:
        server: FL server instance
        clients: List of FL client instances
        properties: List of property names to analyze
        meta_classifier_type: Type of meta-classifier

    Returns:
        Dict mapping property names to attack results

    Example:
        >>> results = analyze_property_leakage(
        ...     server, clients,
        ...     properties=['fraud_rate', 'dataset_size', 'feature_0_mean']
        ... )
        >>> results['fraud_rate']['MAE']
        0.02
    """
    results = {}

    for prop in properties:
        attack, prop_results = execute_attack_on_fl_system(
            server, clients, prop, meta_classifier_type=meta_classifier_type
        )
        results[prop] = prop_results

    return results
