"""
Flower server implementation with defense integration.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.server import ClientManager
from flwr.server.client_proxy import ClientProxy

from ..defenses.base import BaseDefense


class DefendedFedAvg(FedAvg):
    """
    Federated averaging strategy with defense integration.

    Extends Flower's FedAvg to apply robust aggregation defenses
    against Byzantine attacks.
    """

    def __init__(
        self,
        defense: BaseDefense,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 0.5,
        **kwargs,
    ):
        """
        Initialize defended FedAvg strategy.

        Args:
            defense: Defense instance to apply during aggregation
            min_fit_clients: Minimum number of clients for training
            min_available_clients: Minimum number of clients available
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            **kwargs,
        )
        self.defense = defense
        self.defense_metrics_history: List[Dict[str, Any]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates using the configured defense.

        Args:
            server_round: Current round number
            results: List of (client_proxy, fit_result) tuples
            failures: List of failures

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        if not results:
            return None, {}

        # Handle failures
        if failures:
            print(f"Round {server_round}: {len(failures)} failures")

        # Extract client updates
        client_updates = []
        client_metrics = {}

        for client_proxy, fit_res in results:
            # Extract parameters
            parameters = parameters_to_ndarrays(fit_res.parameters)
            client_id = fit_res.metrics.get("client_id", len(client_updates))
            client_updates.append((client_id, np.concatenate([p.flatten() for p in parameters])))

            # Collect metrics
            for key, value in fit_res.metrics.items():
                if key not in client_metrics:
                    client_metrics[key] = []
                client_metrics[key].append(value)

        # Apply defense aggregation
        aggregated_flat = self.defense.defend(client_updates)

        # Convert back to list of arrays
        aggregated_params = self._flat_to_list(aggregated_flat, results[0][1].parameters)

        # Get defense metrics
        defense_metrics = self.defense.get_detection_metrics()
        if defense_metrics:
            self.defense_metrics_history.append({
                "round": server_round,
                **defense_metrics,
            })

        # Combine metrics
        metrics = {}
        for key, values in client_metrics.items():
            if isinstance(values[0], (int, float)):
                metrics[f"{key}_avg"] = float(np.mean(values))

        if defense_metrics:
            metrics.update({f"defense_{k}": v for k, v in defense_metrics.items()})

        # Create parameters object
        parameters = fl.common.ndarrays_to_parameters(aggregated_params)

        return parameters, metrics

    def _flat_to_list(
        self,
        flat_params: np.ndarray,
        reference_params: Parameters,
    ) -> List[np.ndarray]:
        """
        Convert flattened parameters back to list of arrays.

        Args:
            flat_params: Flattened parameter array
            reference_params: Reference parameters for shapes

        Returns:
            List of parameter arrays
        """
        ref_ndarrays = parameters_to_ndarrays(reference_params)
        result = []
        idx = 0

        for ref in ref_ndarrays:
            size = ref.size
            param = flat_params[idx:idx + size].reshape(ref.shape)
            result.append(param)
            idx += size

        return result

    def reset_defense_state(self) -> None:
        """Reset defense state between experiments."""
        self.defense.reset_state()
        self.defense_metrics_history.clear()


def create_server(
    defense: BaseDefense,
    num_rounds: int = 10,
    fraction_fit: float = 0.5,
    min_fit_clients: int = 2,
    min_available_clients: int = 2,
    evaluate_every: int = 1,
) -> DefendedFedAvg:
    """
    Factory function to create a Flower server with defense.

    Args:
        defense: Defense instance to apply during aggregation
        num_rounds: Total number of training rounds
        fraction_fit: Fraction of clients to use for training
        min_fit_clients: Minimum clients for training
        min_available_clients: Minimum available clients
        evaluate_every: Evaluation frequency

    Returns:
        DefendedFedAvg strategy instance
    """
    return DefendedFedAvg(
        defense=defense,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,  # Server-side evaluation not used
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
    )


class ServerConfig:
    """
    Server configuration container.
    """

    def __init__(
        self,
        num_rounds: int = 10,
        fraction_fit: float = 0.5,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        client_resources: Dict[str, int] = None,
    ):
        """
        Initialize server configuration.

        Args:
            num_rounds: Number of training rounds
            fraction_fit: Fraction of clients for training
            min_fit_clients: Minimum clients for training
            min_available_clients: Minimum available clients
            client_resources: Resources per client (num_cpus, num_gpus)
        """
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.client_resources = client_resources or {"num_cpus": 1}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_rounds": self.num_rounds,
            "fraction_fit": self.fraction_fit,
            "min_fit_clients": self.min_fit_clients,
            "min_available_clients": self.min_available_clients,
            "client_resources": self.client_resources,
        }
