"""Metrics computation and logging for federated learning."""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from loguru import logger


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["true_negatives"] = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    metrics["false_positives"] = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    metrics["false_negatives"] = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    metrics["true_positives"] = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    # AUC metrics if probabilities provided
    if y_proba is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
            metrics["auc_pr"] = float(average_precision_score(y_true, y_proba))
        except ValueError as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics["auc_roc"] = 0.0
            metrics["auc_pr"] = 0.0

    return metrics


def compute_fraud_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute fraud-specific metrics.

    Includes business metrics like cost savings.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    # Standard metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)

    # Fraud-specific metrics
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        # Fraud detection rate (recall of fraud class)
        metrics["fraud_detection_rate"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # False alarm rate (FPR)
        metrics["false_alarm_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Cost metrics (assumed costs)
        cost_fn = 1000  # Cost of missing fraud
        cost_fp = 10  # Cost of false alarm
        cost_tp = 10  # Cost of investigating fraud
        cost_tn = 0

        total_cost = (
            fn * cost_fn + fp * cost_fp + tp * cost_tp + tn * cost_tn
        )
        max_cost = (tp + fn) * cost_fn + (fp + tn) * cost_fp

        metrics["total_cost"] = float(total_cost)
        metrics["cost_savings_rate"] = float(1 - total_cost / max_cost) if max_cost > 0 else 0.0

    return metrics


class MetricsLogger:
    """
    Logger for tracking metrics over time.

    Useful for generating reports and visualizations.
    """

    def __init__(self):
        """Initialize metrics logger."""
        self.metrics_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.round_metrics: Dict[int, Dict[str, float]] = {}

    def log(self, round_num: int, metrics: Dict[str, float]) -> None:
        """
        Log metrics for a round.

        Args:
            round_num: Round number
            metrics: Dictionary of metrics
        """
        self.round_metrics[round_num] = metrics.copy()

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append((round_num, value))

    def get_metric(self, metric_name: str) -> List[Tuple[int, float]]:
        """
        Get history of a specific metric.

        Args:
            metric_name: Name of metric

        Returns:
            List of (round, value) tuples
        """
        return self.metrics_history.get(metric_name, [])

    def get_round_metrics(self, round_num: int) -> Dict[str, float]:
        """
        Get all metrics for a specific round.

        Args:
            round_num: Round number

        Returns:
            Dictionary of metrics
        """
        return self.round_metrics.get(round_num, {})

    def get_latest_metrics(self) -> Dict[str, float]:
        """Get metrics from the latest round."""
        if not self.round_metrics:
            return {}

        latest_round = max(self.round_metrics.keys())
        return self.round_metrics[latest_round]

    def compute_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all rounds.

        Returns:
            Dictionary of summary statistics
        """
        summary = {}

        for metric_name, history in self.metrics_history.items():
            if not history:
                continue

            values = [v for _, v in history]

            summary[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "final": float(values[-1]),
                "best": float(np.max(values) if "loss" not in metric_name else np.min(values)),
            }

        return summary

    def get_metrics_dataframe(self):
        """
        Get metrics as a pandas DataFrame.

        Returns:
            DataFrame with metrics over rounds
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not available")
            return None

        if not self.round_metrics:
            return None

        df = pd.DataFrame.from_dict(self.round_metrics, orient="index")
        df.index.name = "round"
        df = df.reset_index()

        return df

    def plot_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot metrics over rounds.

        Args:
            metric_names: List of metrics to plot (all if None)
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return

        if metric_names is None:
            metric_names = ["accuracy", "loss", "f1"]

        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric_name in zip(axes, metric_names):
            history = self.get_metric(metric_name)

            if not history:
                continue

            rounds, values = zip(*history)
            ax.plot(rounds, values, marker="o", label=metric_name)
            ax.set_xlabel("Round")
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f"{metric_name.capitalize()} over Rounds")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved metrics plot to {save_path}")

        plt.close()

    def save_to_file(self, path: str) -> None:
        """
        Save metrics history to file.

        Args:
            path: Path to save metrics (JSON format)
        """
        import json

        data = {
            "metrics_history": {
                k: [[int(r), float(v)] for r, v in v_list]
                for k, v_list in self.metrics_history.items()
            },
            "round_metrics": {
                str(k): {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()}
                for k, v in self.round_metrics.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved metrics to {path}")

    @classmethod
    def load_from_file(cls, path: str) -> "MetricsLogger":
        """
        Load metrics from file.

        Args:
            path: Path to metrics file (JSON format)

        Returns:
            MetricsLogger instance
        """
        import json

        logger = cls()

        with open(path, "r") as f:
            data = json.load(f)

        # Restore metrics history
        for metric_name, history in data["metrics_history"].items():
            logger.metrics_history[metric_name] = [(r, v) for r, v in history]

        # Restore round metrics
        for round_str, metrics in data["round_metrics"].items():
            logger.round_metrics[int(round_str)] = metrics

        return logger


class TrainingTimer:
    """Timer for measuring training time."""

    def __init__(self):
        """Initialize timer."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.round_times: List[Tuple[int, float]] = []

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.time()

    def stop(self) -> float:
        """
        Stop timing and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            logger.warning("Timer was not started")
            return 0.0

        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        return elapsed

    def log_round(self, round_num: int) -> None:
        """Log time for a round."""
        if self.start_time is None:
            self.start()
            return

        elapsed = time.time() - self.start_time
        self.round_times.append((round_num, elapsed))
        self.start_time = time.time()  # Reset for next round

    def get_total_time(self) -> float:
        """Get total training time."""
        return sum(t for _, t in self.round_times)

    def get_avg_round_time(self) -> float:
        """Get average time per round."""
        if not self.round_times:
            return 0.0
        return np.mean([t for _, t in self.round_times])

    def get_round_times(self) -> List[Tuple[int, float]]:
        """Get list of (round, time) tuples."""
        return self.round_times


def log_metrics(
    metrics: Dict[str, float],
    prefix: str = "",
    round_num: Optional[int] = None,
) -> None:
    """
    Log metrics to console.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names
        round_num: Round number (optional)
    """
    round_str = f"Round {round_num}: " if round_num is not None else ""

    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{round_str}{prefix}{key} = {value:.4f}")
        else:
            logger.info(f"{round_str}{prefix}{key} = {value}")
