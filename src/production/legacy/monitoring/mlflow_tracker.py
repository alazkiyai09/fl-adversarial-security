"""MLflow tracking for federated learning experiments."""

from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
import numpy as np
import mlflow
import mlflow.pytorch
from omegaconf import DictConfig
from loguru import logger


class MLflowTracker:
    """
    MLflow tracker for FL experiments.

    Tracks:
    - Hyperparameters
    - Per-round metrics
    - Model checkpoints
    - Privacy metrics (ε, δ)
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking URI (from env if None)
            run_name: Name for this run
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.active_run: Optional[mlflow.ActiveRun] = None

        # Set tracking URI
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name}")
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment: {e}")

    def start_run(self) -> None:
        """Start a new MLflow run."""
        if self.active_run is not None:
            logger.warning("Run already active, not starting new run")
            return

        try:
            self.active_run = mlflow.start_run(run_name=self.run_name)
            logger.info(f"Started MLflow run: {self.active_run.info.run_id}")
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the active MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self.active_run is None:
            logger.warning("No active run to end")
            return

        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
            self.active_run = None
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters
        """
        if self.active_run is None:
            logger.warning("No active run, cannot log params")
            return

        try:
            # Convert parameters to MLflow-compatible types
            mlflow_params = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow_params[key] = value
                elif isinstance(value, (DictConfig, dict)):
                    # Flatten nested dicts
                    for k, v in value.items():
                        if isinstance(v, (str, int, float, bool)):
                            mlflow_params[f"{key}.{k}"] = v
                else:
                    mlflow_params[key] = str(value)

            mlflow.log_params(mlflow_params)
            logger.debug(f"Logged {len(mlflow_params)} parameters")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics
            step: Training step/round number
        """
        if self.active_run is None:
            logger.warning("No active run, cannot log metrics")
            return

        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics at step {step}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_round_metrics(
        self,
        round_num: int,
        metrics: Dict[str, float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log metrics for a federated learning round.

        Args:
            round_num: Round number
            metrics: Dictionary of metrics
            parameters: Optional parameters to log
        """
        # Add round prefix to metrics
        round_metrics = {f"round_{key}": value for key, value in metrics.items()}
        self.log_metrics(round_metrics, step=round_num)

        # Also log without prefix for simpler plotting
        self.log_metrics(metrics, step=round_num)

        # Log parameters if provided (e.g., learning rate changes)
        if parameters:
            param_metrics = {
                f"param_{key}": value for key, value in parameters.items()
                if isinstance(value, (int, float))
            }
            if param_metrics:
                self.log_metrics(param_metrics, step=round_num)

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        **kwargs
    ) -> None:
        """
        Log a PyTorch model to MLflow.

        Args:
            model: PyTorch model
            artifact_path: Path for artifact
            **kwargs: Additional arguments for log_model
        """
        if self.active_run is None:
            logger.warning("No active run, cannot log model")
            return

        try:
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            logger.info(f"Logged model to {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_checkpoint(
        self,
        model: torch.nn.Module,
        round_num: int,
        checkpoint_dir: Path,
    ) -> None:
        """
        Log model checkpoint.

        Args:
            model: PyTorch model
            round_num: Round number
            checkpoint_dir: Checkpoint directory
        """
        if self.active_run is None:
            return

        try:
            checkpoint_path = checkpoint_dir / f"model_round_{round_num}.pt"
            torch.save(model.state_dict(), checkpoint_path)

            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
            logger.debug(f"Logged checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to log checkpoint: {e}")

    def log_hyperparameters(
        self,
        config: DictConfig,
    ) -> None:
        """
        Log configuration hyperparameters.

        Args:
            config: Hydra configuration
        """
        params = {}

        # Flatten config for MLflow
        def flatten_config(cfg: DictConfig, prefix: str = ""):
            for key, value in cfg.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, DictConfig):
                    flatten_config(value, full_key)
                elif isinstance(value, (str, int, float, bool)):
                    params[full_key] = value
                elif isinstance(value, (list, tuple)):
                    params[full_key] = str(value)
                else:
                    params[full_key] = str(value)

        flatten_config(config)
        self.log_params(params)

    def log_privacy_metrics(
        self,
        epsilon: float,
        delta: float,
        round_num: Optional[int] = None,
        **privacy_params
    ) -> None:
        """
        Log differential privacy metrics.

        Args:
            epsilon: Privacy parameter ε
            delta: Privacy parameter δ
            round_num: Current round (optional)
            **privacy_params: Additional privacy parameters
        """
        metrics = {
            "privacy_epsilon": epsilon,
            "privacy_delta": delta,
        }

        metrics.update(privacy_params)

        if round_num is not None:
            metrics["round"] = round_num

        self.log_metrics(metrics, step=round_num)

    def log_client_metrics(
        self,
        round_num: int,
        client_id: int,
        metrics: Dict[str, float],
    ) -> None:
        """
        Log per-client metrics.

        Args:
            round_num: Round number
            client_id: Client ID
            metrics: Client metrics
        """
        client_metrics = {
            f"client_{client_id}_{key}": value
            for key, value in metrics.items()
        }

        self.log_metrics(client_metrics, step=round_num)

    def log_defense_events(
        self,
        round_num: int,
        n_malicious_detected: int,
        n_total_clients: int,
        defense_name: str = "signguard",
    ) -> None:
        """
        Log security defense events.

        Args:
            round_num: Round number
            n_malicious_detected: Number of malicious clients detected
            n_total_clients: Total number of clients
            defense_name: Name of defense mechanism
        """
        metrics = {
            f"defense_{defense_name}_detected": n_malicious_detected,
            f"defense_{defense_name}_ratio": n_malicious_detected / n_total_clients,
        }

        self.log_metrics(metrics, step=round_num)

        if n_malicious_detected > 0:
            logger.warning(
                f"Round {round_num}: {defense_name} detected "
                f"{n_malicious_detected}/{n_total_clients} malicious clients"
            )

    def log_artifact(self, artifact_path: str, artifact_path_optional: Optional[str] = None) -> None:
        """
        Log an artifact (file) to MLflow.

        Args:
            artifact_path: Path to artifact
            artifact_path_optional: Optional MLflow artifact path
        """
        if self.active_run is None:
            return

        try:
            mlflow.log_artifact(artifact_path, artifact_path_optional)
            logger.debug(f"Logged artifact: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def log_figure(self, figure, artifact_file: str) -> None:
        """
        Log a matplotlib figure to MLflow.

        Args:
            figure: Matplotlib figure
            artifact_file: Filename for artifact
        """
        if self.active_run is None:
            return

        try:
            import matplotlib.pyplot as plt
            mlflow.log_figure(figure, artifact_file)
            logger.debug(f"Logged figure: {artifact_file}")
        except Exception as e:
            logger.error(f"Failed to log figure: {e}")

    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if self.active_run is not None:
            return self.active_run.info.run_id
        return None

    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag for the current run.

        Args:
            key: Tag key
            value: Tag value
        """
        if self.active_run is None:
            return

        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.error(f"Failed to set tag: {e}")


def create_mlflow_tracker(config: DictConfig) -> MLflowTracker:
    """
    Create MLflow tracker from configuration.

    Args:
        config: Configuration object

    Returns:
        MLflowTracker instance
    """
    tracker = MLflowTracker(
        experiment_name=config.mlops.mlflow_experiment_name,
        tracking_uri=config.mlops.get("mlflow_tracking_uri", None),
    )
    return tracker
