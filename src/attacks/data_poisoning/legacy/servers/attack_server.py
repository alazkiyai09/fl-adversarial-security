"""
Custom server strategy for tracking attack impact in Federated Learning.

This module implements a custom Flower server strategy that tracks various
metrics to assess the impact of label flipping attacks.
"""

import flwr as fl
from flwr.common import NDArrays, Scalar, Parameters
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fraud_mlp import FraudMLP, set_model_parameters
from metrics.attack_metrics import (
    calculate_per_class_accuracy,
    calculate_convergence_delay,
)


class AttackMetricsStrategy(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy with attack metrics tracking.

    This strategy extends Flower's FedAvg to track:
    - Global model accuracy per round
    - Per-class accuracy (fraud vs legitimate)
    - Number of malicious clients participating
    - Convergence metrics
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[callable] = None,
        initial_parameters: Optional[NDArrays] = None,
        fit_metrics_aggregation_fn: Optional[callable] = None,
        evaluate_metrics_aggregation_fn: Optional[callable] = None,
    ):
        """
        Initialize the AttackMetricsStrategy.

        Args:
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
            eval_fn: Evaluation function for global model evaluation
            initial_parameters: Initial model parameters
            fit_metrics_aggregation_fn: Function to aggregate fit metrics
            evaluate_metrics_aggregation_fn: Function to aggregate evaluation metrics
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        # Track metrics over time
        self.history = defaultdict(list)
        self.round_num = 0
        self.eval_fn = eval_fn

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, fl.common.FitRes]],
        failures: List[Union[Tuple[int, fl.common.FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates and track metrics.

        Args:
            server_round: Current round number
            results: List of (client_id, FitRes) tuples
            failures: List of failed client updates

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        # Count malicious clients
        num_malicious = 0
        client_metrics = []

        for client_id, fit_res in results:
            if fit_res.metrics.get("is_malicious", False):
                num_malicious += 1
            client_metrics.append(fit_res.metrics)

        # Store metrics
        self.history["round"].append(server_round)
        self.history["num_malicious_clients"].append(num_malicious)
        self.history["num_total_clients"].append(len(results))

        # Call parent's aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Add custom metrics
        if aggregated_metrics is None:
            aggregated_metrics = {}

        aggregated_metrics["num_malicious_clients"] = num_malicious
        aggregated_metrics["num_total_clients"] = len(results)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[int, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[int, fl.common.EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation metrics and track attack impact.

        Args:
            server_round: Current round number
            results: List of (client_id, EvaluateRes) tuples
            failures: List of failed client evaluations

        Returns:
            Tuple of (aggregated_loss, metrics)
        """
        # Call parent's aggregate_evaluate
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Store metrics
        if aggregated_loss is not None:
            self.history["centralized_loss"].append(aggregated_loss)

        if aggregated_metrics and "accuracy" in aggregated_metrics:
            self.history["centralized_accuracy"].append(aggregated_metrics["accuracy"])

        # Track per-client accuracies
        client_accuracies = []
        for client_id, eval_res in results:
            if eval_res.metrics:
                client_accuracies.append(eval_res.metrics.get("accuracy", 0.0))

        if client_accuracies:
            avg_client_accuracy = np.mean(client_accuracies)
            self.history["avg_client_accuracy"].append(avg_client_accuracy)

            if aggregated_metrics is None:
                aggregated_metrics = {}
            aggregated_metrics["avg_client_accuracy"] = avg_client_accuracy

        # Perform global model evaluation if eval_fn is provided
        if self.eval_fn is not None and aggregated_parameters := self._get_current_parameters():
            eval_loss, eval_metrics = self.eval_fn(server_round, aggregated_parameters)

            if eval_metrics:
                # Store per-class accuracy
                if "accuracy_fraud" in eval_metrics:
                    self.history["accuracy_fraud"].append(eval_metrics["accuracy_fraud"])

                if "accuracy_legitimate" in eval_metrics:
                    self.history["accuracy_legitimate"].append(eval_metrics["accuracy_legitimate"])

                if "accuracy" in eval_metrics:
                    self.history["global_accuracy"].append(eval_metrics["accuracy"])

                # Update aggregated metrics
                for key, value in eval_metrics.items():
                    aggregated_metrics[key] = value

        self.round_num = server_round

        return aggregated_loss, aggregated_metrics

    def _get_current_parameters(self) -> Optional[NDArrays]:
        """
        Get current aggregated parameters.

        Returns:
            Current model parameters as NDArrays or None
        """
        # This would be set by the framework after aggregate_fit
        # For now, return None as this is handled internally by Flower
        return None

    def get_history(self) -> Dict[str, List]:
        """
        Get the complete history of metrics.

        Returns:
            Dictionary mapping metric names to lists of values
        """
        return dict(self.history)

    def get_convergence_summary(self) -> Dict:
        """
        Get a summary of convergence metrics.

        Returns:
            Dictionary with convergence summary
        """
        summary = {
            "total_rounds": len(self.history.get("round", [])),
            "final_accuracy": self.history.get("global_accuracy", [None])[-1],
            "final_fraud_accuracy": self.history.get("accuracy_fraud", [None])[-1],
            "final_legitimate_accuracy": self.history.get("accuracy_legitimate", [None])[-1],
        }

        # Calculate convergence delay (if baseline available)
        if "global_accuracy" in self.history and len(self.history["global_accuracy"]) > 0:
            final_acc = self.history["global_accuracy"][-1]
            threshold = max(0.01, final_acc * 0.01)  # 1% of final accuracy or 0.01

            for i, acc in enumerate(self.history["global_accuracy"]):
                if abs(acc - final_acc) < threshold:
                    summary["convergence_round"] = i + 1
                    break
            else:
                summary["convergence_round"] = None

        return summary


def create_eval_fn(
    test_loader: torch.utils.data.DataLoader,
    input_size: int = 30,
    device: str = "cpu"
) -> callable:
    """
    Create an evaluation function for the global model.

    Args:
        test_loader: Test data loader
        input_size: Number of input features
        device: Device to evaluate on

    Returns:
        Evaluation function that takes (round, parameters) and returns (loss, metrics)
    """
    model = FraudMLP(input_size=input_size).to(device)
    criterion = nn.CrossEntropyLoss()

    def eval_fn(round_num: int, parameters: NDArrays) -> Tuple[float, Dict[str, Scalar]]:
        """
        Evaluate the global model.

        Args:
            round_num: Current round number
            parameters: Model parameters to evaluate

        Returns:
            Tuple of (loss, metrics)
        """
        # Set parameters
        set_model_parameters(model, parameters)

        # Evaluate
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Per-class tracking
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                logits = model(X)
                loss = criterion(logits, y)

                predictions = torch.argmax(logits, dim=1)

                # Overall metrics
                total_correct += (predictions == y).sum().item()
                total_loss += loss.item() * X.size(0)
                total_samples += X.size(0)

                # Per-class metrics
                for pred, true in zip(predictions.cpu().numpy(), y.cpu().numpy()):
                    class_total[true] += 1
                    if pred == true:
                        class_correct[true] += 1

        # Calculate metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

        # Add per-class accuracy if both classes present
        if 0 in class_total and 1 in class_total:
            accuracy_legitimate = class_correct[0] / class_total[0]
            accuracy_fraud = class_correct[1] / class_total[1]

            metrics["accuracy_legitimate"] = accuracy_legitimate
            metrics["accuracy_fraud"] = accuracy_fraud

        return avg_loss, metrics

    return eval_fn
