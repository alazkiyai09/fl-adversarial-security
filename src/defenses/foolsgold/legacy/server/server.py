"""
Flower server with FoolsGold aggregation.
"""

from typing import Callable, Dict, List, Tuple, Optional, Union
import numpy as np
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.server import Config

from ..aggregators import (
    BaseAggregator,
    FoolsGoldAggregator,
    KrumAggregator,
    MultiKrumAggregator,
    TrimmedMeanAggregator
)


class FoolsGoldStrategy(Strategy):
    """
    Custom strategy supporting various aggregation methods.
    """

    def __init__(
        self,
        aggregator: BaseAggregator,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2
    ):
        """
        Initialize strategy.

        Args:
            aggregator: Aggregation strategy (FoolsGold, Krum, etc.)
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
        """
        super().__init__()
        self.aggregator = aggregator
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        # Track metrics
        self.history_metrics = {
            "loss": [],
            "accuracy": [],
            "aggregation_metrics": []
        }

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: Callable
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for training."""
        # Sample clients
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)

        # Create fit instructions
        config = {
            "num_epochs": 5,
            "learning_rate": 0.01,
            "server_round": server_round
        }

        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: Callable
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure clients for evaluation."""
        # Sample clients
        sample_size, min_num = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)

        # Create evaluate instructions
        config = {"server_round": server_round}
        evaluate_ins = EvaluateIns(parameters, config)

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates using configured aggregator.

        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failures

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        if not results:
            return None, {}

        # Use aggregator
        aggregated_params = self.aggregator.aggregate(results)

        # Compute metrics
        metrics = {}
        losses = [fit_res.metrics.get("loss", 0) for _, fit_res in results]
        if losses:
            metrics["avg_loss"] = float(np.mean(losses))
            metrics["min_loss"] = float(np.min(losses))
            metrics["max_loss"] = float(np.max(losses))

        # Store aggregator-specific metrics
        agg_metrics = self.aggregator.get_metrics()
        metrics["aggregation_metrics"] = agg_metrics

        self.history_metrics["aggregation_metrics"].append(agg_metrics)

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics."""
        if not results:
            return None, {}

        # Average accuracy
        accuracies = [eval_res.metrics.get("accuracy", 0) for _, eval_res in results]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0

        metrics = {"accuracy": avg_accuracy}
        self.history_metrics["accuracy"].append(avg_accuracy)

        return avg_accuracy, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return number of clients for training."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return number of clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


from flwr.common import FitIns, EvaluateIns


def create_foolsgold_server(
    aggregator: BaseAggregator,
    num_rounds: int = 100,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.5
) -> FoolsGoldStrategy:
    """
    Create server strategy with FoolsGold aggregation.

    Args:
        aggregator: Aggregation strategy to use
        num_rounds: Number of training rounds
        fraction_fit: Fraction of clients for training
        fraction_evaluate: Fraction of clients for evaluation

    Returns:
        Configured strategy
    """
    return FoolsGoldStrategy(
        aggregator=aggregator,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2
    )
