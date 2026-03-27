"""Flower client implementation for federated learning."""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from flwr.client import NumPyClient
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from omegaconf import DictConfig
from loguru import logger

from ..models import LSTMFraudDetector, TransformerFraudDetector, XGBoostFraudDetector
from ..privacy.differential_privacy import DPSGDFactory


class FlowerClient(NumPyClient):
    """
    Flower client for federated learning.

    Handles local training and evaluation with optional privacy mechanisms.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        config: DictConfig,
        client_id: int,
        device: torch.device,
    ):
        """
        Initialize Flower client.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            test_loader: Test/evaluation data loader
            config: Configuration object
            client_id: Unique client identifier
            device: Device for training (CPU/GPU)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.client_id = client_id
        self.device = device

        # Training parameters
        self.local_epochs = config.fl.local_epochs
        self.learning_rate = config.model.learning_rate

        # Loss function
        self.criterion = self._get_criterion()

        # Optimizer
        self.optimizer = self._get_optimizer()

        # Differential Privacy
        self.dp_factory = None
        if config.privacy.dp_enabled:
            self.dp_factory = DPSGDFactory(
                noise_multiplier=config.privacy.noise_multiplier,
                max_grad_norm=config.privacy.max_grad_norm,
                delta=config.privacy.delta,
                epsilon=config.privacy.epsilon,
            )
            # Replace optimizer with DP version
            self.optimizer = self.dp_factory.create_dp_optimizer(
                self.model.parameters(), self.learning_rate
            )
            logger.info(f"Client {client_id}: DP-SGD enabled")

        # Learning rate scheduler
        self.scheduler = self._get_scheduler()

        # Metrics tracking
        self.round_metrics: Dict[int, Dict[str, float]] = {}

        logger.info(
            f"Client {client_id} initialized: "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(test_loader.dataset)} test samples"
        )

    def _get_criterion(self) -> nn.Module:
        """Get loss function."""
        loss_fn = self.config.model.get("loss_function", "cross_entropy")

        if loss_fn == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_fn == "focal":
            from ..models.focal_loss import FocalLoss
            return FocalLoss(alpha=1, gamma=2)
        else:
            return nn.CrossEntropyLoss()

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer."""
        optimizer_name = self.config.model.get("optimizer", "adam")
        weight_decay = self.config.model.get("weight_decay", 1e-5)

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler."""
        scheduler_type = self.config.model.get("scheduler", "none")

        if scheduler_type == "step":
            step_size = self.config.model.get("step_size", 20)
            gamma = self.config.model.get("gamma", 0.5)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.local_epochs
            )
        elif scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            return None

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """
        Get current model parameters.

        Args:
            config: Server configuration

        Returns:
            List of parameter arrays
        """
        logger.debug(f"Client {self.client_id}: get_parameters called")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters.

        Args:
            parameters: List of parameter arrays from server
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        logger.debug(f"Client {self.client_id}: Parameters updated from server")

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train model locally.

        Args:
            parameters: Current global model parameters
            config: Server configuration (round number, etc.)

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Set current global parameters
        self.set_parameters(parameters)

        # Get round number
        round_num = config.get("round", 0)
        logger.info(f"Client {self.client_id}: Starting training for round {round_num}")

        # Train for local epochs
        train_loss, train_metrics = self._train_round(round_num)

        # Get updated parameters
        updated_params = self.get_parameters(config={})

        # Compute metrics
        num_examples = len(self.train_loader.dataset)
        metrics = {
            "loss": float(train_loss),
            "accuracy": float(train_metrics["accuracy"]),
            "precision": float(train_metrics.get("precision", 0.0)),
            "recall": float(train_metrics.get("recall", 0.0)),
            "f1": float(train_metrics.get("f1", 0.0)),
            "client_id": int(self.client_id),
        }

        # Add DP metrics if applicable
        if self.dp_factory is not None:
            metrics["epsilon"] = float(self.dp_factory.epsilon)
            metrics["dp_enabled"] = True

        logger.info(
            f"Client {self.client_id}: Round {round_num} - "
            f"Loss: {train_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}"
        )

        # Store metrics
        self.round_metrics[round_num] = metrics

        return updated_params, num_examples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model locally.

        Args:
            parameters: Current global model parameters
            config: Server configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set parameters
        self.set_parameters(parameters)

        # Evaluate
        test_loss, test_metrics = self._evaluate()

        num_examples = len(self.test_loader.dataset)
        metrics = {
            "loss": float(test_loss),
            "accuracy": float(test_metrics["accuracy"]),
            "precision": float(test_metrics.get("precision", 0.0)),
            "recall": float(test_metrics.get("recall", 0.0)),
            "f1": float(test_metrics.get("f1", 0.0)),
            "auc_roc": float(test_metrics.get("auc_roc", 0.0)),
        }

        logger.debug(
            f"Client {self.client_id}: Evaluation - "
            f"Loss: {test_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}"
        )

        return float(test_loss), num_examples, metrics

    def _train_round(self, round_num: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one round.

        Args:
            round_num: Current round number

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_preds = []
            epoch_labels = []

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()

                # Gradient clipping (for DP or general stability)
                if self.dp_factory is not None:
                    # DP-SGD handles clipping internally
                    pass
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                self.optimizer.step()

                epoch_loss += loss.item()
                preds = output.argmax(dim=1).cpu()
                epoch_preds.extend(preds.numpy())
                epoch_labels.extend(target.cpu().numpy())

            # Epoch metrics
            epoch_loss /= len(self.train_loader)
            epoch_metrics = self._compute_metrics(epoch_preds, epoch_labels)

            logger.debug(
                f"Client {self.client_id} - Round {round_num}, Epoch {epoch + 1}/{self.local_epochs}: "
                f"Loss: {epoch_loss:.4f}, Acc: {epoch_metrics['accuracy']:.4f}"
            )

            total_loss += epoch_loss
            all_preds.extend(epoch_preds)
            all_labels.extend(epoch_labels)

            # Step scheduler if applicable
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_loss)
                else:
                    self.scheduler.step()

        avg_loss = total_loss / self.local_epochs
        metrics = self._compute_metrics(all_preds, all_labels)

        return avg_loss, metrics

    def _evaluate(self) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on test set.

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                preds = output.argmax(dim=1).cpu()
                probs = torch.softmax(output, dim=1)[:, 1].cpu()

                all_preds.extend(preds.numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.numpy())

        avg_loss = total_loss / len(self.test_loader)
        metrics = self._compute_metrics(all_preds, all_labels, all_probs)

        return avg_loss, metrics

    def _compute_metrics(
        self,
        preds: List[int],
        labels: List[int],
        probs: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            preds: Predictions
            labels: True labels
            probs: Predicted probabilities (optional, for AUC)

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

        # Compute AUC if probabilities provided
        if probs is not None:
            from sklearn.metrics import roc_auc_score
            try:
                metrics["auc_roc"] = roc_auc_score(labels, probs)
            except ValueError:
                metrics["auc_roc"] = 0.0

        return metrics

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "client_id": self.client_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "round_metrics": self.round_metrics,
        }, path)

        logger.info(f"Client {self.client_id}: Model saved to {path}")

    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.round_metrics = checkpoint.get("round_metrics", {})

        logger.info(f"Client {self.client_id}: Model loaded from {path}")


def create_client(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: DictConfig,
    client_id: int,
) -> FlowerClient:
    """
    Create a Flower client.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration
        client_id: Client identifier

    Returns:
        FlowerClient instance
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    client = FlowerClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        client_id=client_id,
        device=device,
    )

    return client
