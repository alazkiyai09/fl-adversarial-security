"""
Metrics for evaluating attack success and model performance.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)


def compute_attack_success_rate(
    model: nn.Module,
    test_loader: DataLoader,
    target_class: int = 0,
    attack_type: str = "label_flip",
    device: str = "cpu",
) -> float:
    """
    Compute Attack Success Rate (ASR).

    For label flip attacks: fraction of samples from source class
    misclassified as target class.
    For backdoor attacks: fraction of samples with trigger
    misclassified as target class.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        target_class: Target class for attack (default: 0)
        attack_type: Type of attack ('label_flip', 'backdoor')
        device: Device for computation

    Returns:
        Attack success rate as float in [0, 1]
    """
    model.eval()
    model.to(device)

    successful_attacks = 0
    total_relevant = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)

            if attack_type == "label_flip":
                # For label flip, measure how many non-target samples
                # are classified as target
                source_mask = y_batch != target_class
                if source_mask.sum() > 0:
                    relevant_predicted = predicted[source_mask]
                    successful_attacks += (relevant_predicted == target_class).sum().item()
                    total_relevant += source_mask.sum().item()

            elif attack_type == "backdoor":
                # For backdoor, measure success on all samples
                # (in real scenario, would add trigger first)
                successful_attacks += (predicted == target_class).sum().item()
                total_relevant += len(y_batch)

    return successful_attacks / total_relevant if total_relevant > 0 else 0.0


def compute_auprc(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """
    Compute Area Under Precision-Recall Curve (AUPRC).

    Important metric for imbalanced datasets like fraud detection.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device for computation

    Returns:
        AUPRC as float in [0, 1]
    """
    model.eval()
    model.to(device)

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)

            outputs = model(x_batch)
            probabilities = torch.softmax(outputs, dim=1)

            # Get scores for positive class (class 1 - fraud)
            all_scores.append(probabilities[:, 1].cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(labels, scores)

    # Compute AUPRC
    auprc = auc(recall, precision)

    return float(auprc)


def compute_clean_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """
    Compute clean accuracy on test data.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device for computation

    Returns:
        Accuracy as float in [0, 1]
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == y_batch).sum().item()
            total += len(y_batch)

    return correct / total if total > 0 else 0.0


def compute_fraud_detection_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute comprehensive fraud detection metrics.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device for computation

    Returns:
        Dictionary with metrics:
            - accuracy: Overall accuracy
            - precision: Precision for fraud class
            - recall: Recall for fraud class
            - f1: F1 score for fraud class
            - auprc: Area under PR curve
            - confusion_matrix: Confusion matrix as list
    """
    model.eval()
    model.to(device)

    all_predictions = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)

            outputs = model(x_batch)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_scores.append(probabilities[:, 1].cpu().numpy())

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)

    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Compute precision, recall, f1
    tp = cm[1, 1]  # True positives
    fp = cm[0, 1]  # False positives
    fn = cm[1, 0]  # False negatives
    tn = cm[0, 0]  # True negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Compute AUPRC
    pr_precision, pr_recall, _ = precision_recall_curve(labels, scores)
    auprc = auc(pr_recall, pr_precision)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auprc": float(auprc),
        "confusion_matrix": cm.tolist(),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def compute_model_size(model: nn.Module) -> Dict[str, int]:
    """
    Compute model size metrics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with size metrics:
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            - model_size_mb: Model size in megabytes
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size in bytes (assuming float32)
    model_size_bytes = total_params * 4
    model_size_mb = model_size_bytes / (1024 * 1024)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
    }


class MetricsHistory:
    """
    Container for tracking metrics across training rounds.
    """

    def __init__(self):
        """Initialize metrics history."""
        self.history: Dict[str, List[float]] = {
            "round": [],
            "train_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "auprc": [],
            "asr": [],
            "communication_cost": [],
        }

    def add_metrics(
        self,
        round_num: int,
        train_loss: float,
        train_accuracy: float,
        test_accuracy: float,
        auprc: float,
        asr: float,
        communication_cost: float = 0.0,
    ) -> None:
        """
        Add metrics for a round.

        Args:
            round_num: Round number
            train_loss: Training loss
            train_accuracy: Training accuracy
            test_accuracy: Test accuracy
            auprc: Area under PR curve
            asr: Attack success rate
            communication_cost: Communication cost in bytes
        """
        self.history["round"].append(round_num)
        self.history["train_loss"].append(train_loss)
        self.history["train_accuracy"].append(train_accuracy)
        self.history["test_accuracy"].append(test_accuracy)
        self.history["auprc"].append(auprc)
        self.history["asr"].append(asr)
        self.history["communication_cost"].append(communication_cost)

    def get_final_metrics(self) -> Dict[str, float]:
        """
        Get final metrics from last round.

        Returns:
            Dictionary with final metrics
        """
        if len(self.history["round"]) == 0:
            return {}

        return {
            key: values[-1] if values else 0.0
            for key, values in self.history.items()
        }

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert history to dictionary."""
        return self.history.copy()

    def get_metric_series(self, metric_name: str) -> np.ndarray:
        """
        Get time series for a specific metric.

        Args:
            metric_name: Name of metric

        Returns:
            Array of metric values over time
        """
        return np.array(self.history.get(metric_name, []))

    def compute_convergence_round(
        self,
        metric: str = "test_accuracy",
        threshold: float = 0.01,
        window: int = 3,
    ) -> Optional[int]:
        """
        Compute round where model converged.

        Convergence is defined as when the metric change is below threshold
        for a consecutive window of rounds.

        Args:
            metric: Metric to monitor
            threshold: Convergence threshold
            window: Number of consecutive rounds below threshold

        Returns:
            Convergence round or None if not converged
        """
        values = self.history.get(metric, [])
        if len(values) < window:
            return None

        for i in range(len(values) - window):
            window_values = values[i:i + window]
            if max(window_values) - min(window_values) < threshold:
                return i

        return None
