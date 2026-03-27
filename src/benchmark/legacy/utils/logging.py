"""
MLflow integration for experiment tracking and logging.
"""

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from typing import Any, Dict, Optional, List
from pathlib import Path
import json


class MLflowLogger:
    """
    MLflow logger for tracking experiments.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (default: ./mlruns)
            run_name: Optional run name
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)
        self.active = True

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters.

        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            # Convert complex types to strings
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            mlflow.log_param(key, str(value))

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional training step/round
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional training step/round
        """
        mlflow.log_metric(key, value, step=step)

    def log_model(
        self, model: torch.nn.Module, artifact_path: str = "model"
    ) -> None:
        """
        Log PyTorch model.

        Args:
            model: PyTorch model to log
            artifact_path: Path for artifact
        """
        mlflow.pytorch.log_model(model, artifact_path)

    def log_dict(self, data: Dict[str, Any], filename: str) -> None:
        """
        Log dictionary as JSON artifact.

        Args:
            data: Dictionary to log
            filename: Artifact filename
        """
        temp_path = Path(f"/tmp/{filename}")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(str(temp_path))
        temp_path.unlink()

    def log_numpy(self, array: np.ndarray, filename: str) -> None:
        """
        Log numpy array as artifact.

        Args:
            array: Numpy array to log
            filename: Artifact filename
        """
        temp_path = Path(f"/tmp/{filename}")
        np.save(temp_path, array)
        mlflow.log_artifact(str(temp_path))
        temp_path.unlink()

    def log_figure(self, fig, filename: str) -> None:
        """
        Log matplotlib figure as artifact.

        Args:
            fig: Matplotlib figure
            filename: Artifact filename
        """
        temp_path = Path(f"/tmp/{filename}")
        fig.savefig(temp_path, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(str(temp_path))
        temp_path.unlink()

    def log_artifact(self, local_path: str) -> None:
        """
        Log local file as artifact.

        Args:
            local_path: Path to local file
        """
        mlflow.log_artifact(local_path)

    def set_tag(self, key: str, value: str) -> None:
        """
        Set tag for run.

        Args:
            key: Tag key
            value: Tag value
        """
        mlflow.set_tag(key, value)

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self.active:
            mlflow.end_run(status=status)
            self.active = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")
        return False

    def get_run_id(self) -> str:
        """Get current run ID."""
        return self.run.info.run_id
