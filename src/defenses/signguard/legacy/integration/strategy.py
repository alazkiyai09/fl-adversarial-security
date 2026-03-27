"""
SignGuard Strategy for Flower Framework

Custom Flower strategy integrating all SignGuard components.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Scalar
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from .server import SignGuardServer


class SignGuardStrategy(Strategy):
    """
    Flower strategy with SignGuard defense.

    Orchestrates:
    - Signature verification
    - Anomaly detection
    - Reputation updates
    - Weighted aggregation
    """

    def __init__(self,
                 config: Optional[dict] = None,
                 initial_parameters: Optional[List[np.ndarray]] = None,
                 min_fit_clients: int = 5,
                 min_evaluate_clients: int = 3,
                 min_available_clients: int = 5,
                 fraction_fit: float = 0.5,
                 fraction_evaluate: float = 0.2):
        """
        Initialize SignGuardStrategy.

        Args:
            config: Configuration dictionary
            initial_parameters: Initial global model parameters
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
        """
        super().__init__()

        # Configuration
        self.config = config or {}

        # Flower strategy parameters
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate

        # SignGuard server (handles all defense logic)
        self.server = SignGuardServer(config)

        # Global model
        if initial_parameters is not None:
            self.server.set_global_model(initial_parameters)
        self.initial_parameters = initial_parameters

        # Round tracking
        self.current_round = 0

    def initialize_parameters(self,
                              client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """
        Initialize global model parameters.

        Args:
            client_manager: Flower client manager

        Returns:
            Initial parameters or None
        """
        if self.initial_parameters is not None:
            return ndarrays_to_parameters(self.initial_parameters)
        return None

    def configure_fit(self,
                     server_round: int,
                     parameters: Parameters,
                     client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure clients for training round.

        Args:
            server_round: Current round number
            parameters: Global model parameters
            client_manager: Flower client manager

        Returns:
            List of (client, fit_ins) tuples
        """
        # Update round
        self.current_round = server_round
        self.server.current_round = server_round

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available(),
            min_fit_clients=self.min_fit_clients
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )

        # Convert parameters to numpy
        config = {
            'server_round': server_round,
            'local_epochs': self.config.get('federated_learning', {}).get('local_epochs', 5)
        }

        # Create fit instructions
        fit_ins = fl.common.FitIns(parameters, config)

        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self,
                          server_round: int,
                          parameters: Parameters,
                          client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """
        Configure clients for evaluation round.

        Args:
            server_round: Current round number
            parameters: Global model parameters
            client_manager: Flower client manager

        Returns:
            List of (client, evaluate_ins) tuples
        """
        # Sample clients for evaluation
        if self.fraction_evaluate == 0.0:
            return []

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available(),
            min_evaluate_clients=self.min_evaluate_clients
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )

        # Create evaluate instructions
        evaluate_ins = fl.common.EvaluateIns(parameters, {})

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(self,
                     server_round: int,
                     results: List[Tuple[ClientProxy, FitRes]],
                     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates with SignGuard defense.

        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failed clients

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        # Handle failures
        if not results:
            return None, {}

        # Convert results to signed updates
        signed_updates = []
        for client_proxy, fit_res in results:
            # Extract client ID from properties
            client_id = fit_res.metrics.get('client_id', client_proxy.cid)

            # Convert parameters to numpy
            parameters = parameters_to_ndarrays(fit_res.parameters)

            # Extract signature and metadata
            signature_hex = fit_res.metrics.get('signature', '')
            timestamp = fit_res.metrics.get('timestamp', time.time())
            num_examples = fit_res.num_examples

            signed_updates.append((
                client_id,
                parameters,
                signature_hex,
                timestamp,
                num_examples
            ))

        # Process round with SignGuard server
        aggregated_params, round_metadata = self.server.process_round(signed_updates)

        # Convert back to Flower parameters
        aggregated_parameters = ndarrays_to_parameters(aggregated_params)

        # Compute metrics
        metrics = {
            'round': server_round,
            'num_clients': len(results),
            'valid_clients': round_metadata['valid_clients'],
            'num_anomalous': round_metadata['num_anomalous'],
            'mean_anomaly_score': round_metadata['mean_anomaly_score'],
        }

        # Add reputation stats
        rep_stats = self.server.get_reputation_stats()
        metrics['reputation_mean'] = rep_stats['mean']
        metrics['reputation_std'] = rep_stats['std']
        metrics['reputation_min'] = rep_stats['min']
        metrics['reputation_max'] = rep_stats['max']

        print(f"\nRound {server_round} metrics: {metrics}")

        return aggregated_parameters, metrics

    def aggregate_evaluate(self,
                          server_round: int,
                          results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
                          failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation metrics.

        Args:
            server_round: Current round number
            results: List of (client, evaluate_result) tuples
            failures: List of failed clients

        Returns:
            Tuple of (loss, metrics)
        """
        if not results:
            return None, {}

        # Compute weighted average of metrics
        total_examples = sum(eval_res.num_examples for _, eval_res in results)
        weighted_loss = sum(
            eval_res.num_examples * eval_res.loss
            for _, eval_res in results
        ) / total_examples if total_examples > 0 else 0.0

        # Aggregate accuracy
        accuracies = [eval_res.metrics.get('accuracy', 0.0) for _, eval_res in results]
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0

        metrics = {
            'accuracy': avg_accuracy,
            'num_clients': len(results)
        }

        return weighted_loss, metrics

    def evaluate(self,
                server_round: int,
                parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate global model parameters.

        Args:
            server_round: Current round number
            parameters: Global model parameters

        Returns:
            Tuple of (loss, metrics) or None
        """
        # This method is required by Flower Strategy but we don't use it
        # because we use aggregate_evaluate instead
        return None

    def num_fit_clients(self, num_available_clients: int, min_fit_clients: int) -> Tuple[int, int]:
        """
        Determine number of clients for training.

        Args:
            num_available_clients: Total available clients
            min_fit_clients: Minimum clients required

        Returns:
            Tuple of (sample_size, min_clients)
        """
        sample_size = int(min(num_available_clients, max(min_fit_clients, self.fraction_fit * num_available_clients)))
        return sample_size, min_fit_clients

    def num_evaluation_clients(self, num_available_clients: int, min_evaluate_clients: int) -> Tuple[int, int]:
        """
        Determine number of clients for evaluation.

        Args:
            num_available_clients: Total available clients
            min_evaluate_clients: Minimum clients required

        Returns:
            Tuple of (sample_size, min_clients)
        """
        if self.fraction_evaluate == 0.0:
            return 0, min_evaluate_clients

        sample_size = int(min(num_available_clients, max(min_evaluate_clients, self.fraction_evaluate * num_available_clients)))
        return sample_size, min_evaluate_clients

    def get_server(self) -> SignGuardServer:
        """Get SignGuard server instance."""
        return self.server

    def get_reputations(self) -> Dict[str, float]:
        """Get all client reputations."""
        return self.server.get_reputations()

    def get_detection_history(self) -> List[Dict]:
        """Get detection history."""
        return self.server.get_detection_history()

    def get_aggregation_history(self) -> List[Dict]:
        """Get aggregation history."""
        return self.server.get_aggregation_history()


def create_signguard_strategy(config: Optional[dict] = None,
                              initial_parameters: Optional[List[np.ndarray]] = None) -> SignGuardStrategy:
    """
    Factory function to create SignGuard strategy.

    Args:
        config: Configuration dictionary
        initial_parameters: Initial global model parameters

    Returns:
        SignGuardStrategy instance
    """
    # Extract FL config
    fl_config = config.get('federated_learning', {})

    return SignGuardStrategy(
        config=config,
        initial_parameters=initial_parameters,
        min_fit_clients=fl_config.get('clients_per_round', 5),
        min_evaluate_clients=max(1, fl_config.get('clients_per_round', 5) // 3),
        fraction_fit=fl_config.get('fraction_fit', 0.5),
        fraction_evaluate=fl_config.get('fraction_evaluate', 0.2)
    )
