"""Flower server implementation for federated learning."""

from typing import Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import multiprocessing as mp

import torch
import numpy as np
from flwr.server import ServerConfig, Server, ClientManager
from flwr.server.strategy import Strategy
from flwr.common import Parameters, Scalar
from omegaconf import DictConfig
from loguru import logger

from .strategy import create_strategy
from .client import FlowerClient, create_client
from ..monitoring.mlflow_tracker import MLflowTracker
from ..utils import get_device, get_fl_logger


class FlowerServer:
    """
    Flower server for federated learning.

    Manages the training process across multiple clients with
    support for privacy mechanisms and defenses.
    """

    def __init__(
        self,
        config: DictConfig,
        strategy: Optional[Strategy] = None,
    ):
        """
        Initialize Flower server.

        Args:
            config: Configuration object
            strategy: Custom strategy (created from config if None)
        """
        self.config = config
        self.fl_config = config.fl

        # Create strategy if not provided
        if strategy is None:
            strategy = create_strategy(
                strategy_name=config.strategy.name,
                config=config,
                defense_config=config.security,
            )

        self.strategy = strategy

        # Server configuration
        self.server_address = f"{self.fl_config.server_address}:{self.fl_config.server_port}"
        self.n_rounds = self.fl_config.n_rounds

        # MLflow tracking
        self.mlflow_tracker: Optional[MLflowTracker] = None
        if config.mlops.mlflow_enabled:
            self.mlflow_tracker = MLflowTracker(
                experiment_name=config.mlops.mlflow_experiment_name,
            )

        # Logging
        self.logger = get_fl_logger("server")

        # Model checkpointing
        self.checkpoint_dir = Path(self.fl_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._current_round = 0

        self.logger.info(f"Flower server initialized for {self.n_rounds} rounds")

    def start(
        self,
        client_fn: Callable[[str], FlowerClient],
        num_clients: Optional[int] = None,
    ) -> None:
        """
        Start the federated learning server.

        Args:
            client_fn: Function to create clients
            num_clients: Number of clients (from config if None)
        """
        if num_clients is None:
            num_clients = self.config.data.n_clients

        self.logger.info(f"Starting FL server with {num_clients} clients")

        # Start MLflow run
        if self.mlflow_tracker:
            self.mlflow_tracker.start_run()
            self.mlflow_tracker.log_params(self._get_config_params())

        # Create server configuration
        server_config = ServerConfig(
            num_rounds=self.n_rounds,
            fraction_fit=self.fl_config.get("fraction_fit", 1.0),
            fraction_evaluate=self.fl_config.get("fraction_evaluate", 1.0),
            min_fit_clients=self.fl_config.min_fit_clients,
            min_evaluate_clients=self.fl_config.min_evaluate_clients,
            min_available_clients=self.fl_config.min_available_clients,
        )

        # Start server
        try:
            # For simulation, use start_server
            from flwr.server import start_server

            history = start_server(
                server_address=self.server_address,
                config=server_config,
                strategy=self.strategy,
                client_manager=ClientManager(),
            )

            self.logger.info("Federated learning completed")

            # Log final results
            if self.mlflow_tracker:
                self._log_final_results(history)

        except KeyboardInterrupt:
            self.logger.warning("Server stopped by user")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
        finally:
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run()

    def _get_config_params(self) -> Dict[str, Scalar]:
        """Get configuration parameters for logging."""
        params = {
            "n_rounds": self.n_rounds,
            "n_clients": self.config.data.n_clients,
            "local_epochs": self.fl_config.local_epochs,
            "batch_size": self.config.data.batch_size,
            "learning_rate": self.config.model.learning_rate,
            "model_type": self.config.model.type,
            "strategy": self.config.strategy.name,
        }

        # Privacy parameters
        if self.config.privacy.dp_enabled:
            params.update({
                "dp_enabled": True,
                "epsilon": self.config.privacy.epsilon,
                "delta": self.config.privacy.delta,
                "noise_multiplier": self.config.privacy.noise_multiplier,
                "max_grad_norm": self.config.privacy.max_grad_norm,
            })

        if self.config.privacy.secure_agg_enabled:
            params["secure_aggregation"] = True

        # Security parameters
        if self.config.security.signguard_enabled:
            params.update({
                "signguard_enabled": True,
                "signguard_threshold": self.config.security.signguard_threshold,
            })

        return params

    def _log_final_results(self, history) -> None:
        """Log final training results to MLflow."""
        if not self.mlflow_tracker:
            return

        # Extract metrics from history
        if hasattr(history, "losses_centralized"):
            losses = history.losses_centralized
            if losses:
                final_loss = losses[-1][1]  # (round, loss)
                self.mlflow_tracker.log_metric("final_loss", final_loss)

        if hasattr(history, "metrics_centralized"):
            metrics = history.metrics_centralized
            if "accuracy" in metrics:
                accuracies = metrics["accuracy"]
                if accuracies:
                    final_accuracy = accuracies[-1][1]
                    self.mlflow_tracker.log_metric("final_accuracy", final_accuracy)

        self.logger.info("Final results logged to MLflow")

    def save_checkpoint(self, round_num: int, parameters: Parameters) -> None:
        """
        Save model checkpoint.

        Args:
            round_num: Current round number
            parameters: Model parameters
        """
        checkpoint_path = self.checkpoint_dir / f"model_round_{round_num}.pt"

        # Convert parameters to numpy arrays
        ndarrays = parameters_to_ndarrays(parameters)

        torch.save({
            "round": round_num,
            "parameters": ndarrays,
            "config": self.config,
        }, checkpoint_path)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> Parameters:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Model parameters
        """
        checkpoint = torch.load(checkpoint_path)
        ndarrays = checkpoint["parameters"]
        parameters = ndarrays_to_parameters(ndarrays)

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return parameters

    def evaluate_global_model(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate global model on test set.

        Args:
            model: Global model
            test_loader: Test data loader

        Returns:
            Dictionary of metrics
        """
        model.eval()
        device = get_device(self.config)
        model = model.to(device)

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0

        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                preds = output.argmax(dim=1).cpu()
                probs = torch.softmax(output, dim=1)[:, 1].cpu()

                all_preds.extend(preds.numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.numpy())

        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            "loss": total_loss / len(test_loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }

        try:
            metrics["auc_roc"] = roc_auc_score(all_labels, all_probs)
        except ValueError:
            metrics["auc_roc"] = 0.0

        return metrics


class SimulationServer:
    """
    Server for local FL simulation (all clients in one process).

    Useful for testing and development without network communication.
    """

    def __init__(
        self,
        config: DictConfig,
        clients: List[FlowerClient],
    ):
        """
        Initialize simulation server.

        Args:
            config: Configuration object
            clients: List of Flower clients
        """
        self.config = config
        self.clients = clients
        self.n_rounds = config.fl.n_rounds

        # Create strategy
        self.strategy = create_strategy(
            strategy_name=config.strategy.name,
            config=config,
            defense_config=config.security,
        )

        # MLflow tracking
        self.mlflow_tracker: Optional[MLflowTracker] = None
        if config.mlops.mlflow_enabled:
            self.mlflow_tracker = MLflowTracker(
                experiment_name=config.mlops.mlflow_experiment_name,
            )

        self.logger = get_fl_logger("sim_server")
        self.logger.info(f"Simulation server initialized with {len(clients)} clients")

    def run(self) -> Dict[str, List[float]]:
        """
        Run federated learning simulation.

        Returns:
            Dictionary of metrics over rounds
        """
        if self.mlflow_tracker:
            self.mlflow_tracker.start_run()
            self.mlflow_tracker.log_params(self._get_config_params())

        # Initialize global model
        global_params = self._initialize_parameters()

        # Track metrics
        history = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

        for round_num in range(1, self.n_rounds + 1):
            self.logger.info(f"=== Round {round_num}/{self.n_rounds} ===")

            # Select clients for this round
            num_clients = self.config.fl.min_fit_clients
            selected_clients = np.random.choice(
                self.clients, size=num_clients, replace=False
            )

            # Client training
            fit_results = []
            for client in selected_clients:
                client_params = [p.cpu().numpy() for p in client.model.parameters()]

                fit_res = client.fit(
                    parameters=client_params,
                    config={"round": round_num},
                )
                fit_results.append(fit_res)

            # Aggregate updates
            aggregated_params, aggregate_metrics = self._aggregate(fit_results)

            # Update all clients with new global model
            for client in self.clients:
                client.set_parameters(aggregated_params)

            # Evaluate global model
            eval_metrics = self._evaluate_global(aggregated_params)

            # Log metrics
            for key, value in eval_metrics.items():
                history[key].append(value)

            if self.mlflow_tracker:
                self.mlflow_tracker.log_round_metrics(
                    round_num=round_num,
                    metrics=eval_metrics,
                    parameters=self._get_config_params(),
                )

            self.logger.info(
                f"Round {round_num} - "
                f"Loss: {eval_metrics['loss']:.4f}, "
                f"Accuracy: {eval_metrics['accuracy']:.4f}"
            )

        if self.mlflow_tracker:
            self.mlflow_tracker.end_run()

        self.logger.info("Simulation completed")
        return history

    def _initialize_parameters(self) -> List[np.ndarray]:
        """Initialize global model parameters."""
        # Use parameters from first client
        return [p.cpu().numpy() for p in self.clients[0].model.parameters()]

    def _aggregate(
        self, fit_results: List[Tuple[List[np.ndarray], int, Dict]]
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Aggregate client updates.

        Args:
            fit_results: List of (parameters, num_examples, metrics) tuples

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        # Simple weighted average
        total_examples = sum(num_examples for _, num_examples, _ in fit_results)

        aggregated_params = []
        num_layers = len(fit_results[0][0])

        for layer_idx in range(num_layers):
            layer_params = []
            weights = []

            for params, num_examples, _ in fit_results:
                layer_params.append(params[layer_idx])
                weights.append(num_examples)

            # Weighted average
            weighted_params = np.average(layer_params, axis=0, weights=weights)
            aggregated_params.append(weighted_params)

        # Aggregate metrics
        metrics = {}
        for _, _, client_metrics in fit_results:
            for key, value in client_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = 0.0
                    metrics[key] += value

        # Average metrics
        for key in metrics:
            metrics[key] /= len(fit_results)

        return aggregated_params, metrics

    def _evaluate_global(self, parameters: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate global model using all clients' test data."""
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for client in self.clients:
            # Set parameters
            client.set_parameters(parameters)

            # Evaluate
            loss, metrics = client._evaluate()
            total_loss += loss

            # Get predictions
            client.model.eval()
            with torch.no_grad():
                for data, target in client.test_loader:
                    data = data.to(client.device)
                    output = client.model(data)
                    preds = output.argmax(dim=1).cpu()
                    all_preds.extend(preds.numpy())
                    all_labels.extend(target.numpy())

        # Compute overall metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        avg_loss = total_loss / len(self.clients)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }

        return metrics

    def _get_config_params(self) -> Dict[str, Scalar]:
        """Get configuration parameters for logging."""
        return {
            "n_rounds": self.n_rounds,
            "n_clients": len(self.clients),
            "local_epochs": self.config.fl.local_epochs,
            "model_type": self.config.model.type,
            "strategy": self.config.strategy.name,
        }


def create_server(config: DictConfig) -> FlowerServer:
    """
    Create a Flower server.

    Args:
        config: Configuration object

    Returns:
        FlowerServer instance
    """
    server = FlowerServer(config=config)
    return server


def start_simulation(
    config: DictConfig,
    clients: List[FlowerClient],
) -> Dict[str, List[float]]:
    """
    Start FL simulation.

    Args:
        config: Configuration object
        clients: List of Flower clients

    Returns:
        Training history
    """
    server = SimulationServer(config=config, clients=clients)
    history = server.run()
    return history


# Import utility function
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
