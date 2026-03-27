"""
Federated Learning Client - Local training on client data.

This module implements FL clients that train local models on their
private data and submit updates to the server.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset

from .model import FraudDetectionModel, create_model


class FederatedClient:
    """Federated learning client for fraud detection."""

    def __init__(
        self,
        client_id: int,
        dataset: pd.DataFrame,
        model_config: Dict[str, Any],
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer_type: str = 'adam'
    ):
        """Initialize federated client.

        Args:
            client_id: Unique client identifier
            dataset: Local training data (must include 'label' column)
            model_config: Configuration for model creation
            local_epochs: Number of local training epochs
            batch_size: Batch size for local training
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer ('adam', 'sgd')
        """
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        # Extract features and labels
        self.feature_cols = [col for col in dataset.columns if col != 'label']
        self.X = dataset[self.feature_cols].values.astype(np.float32)
        self.y = dataset['label'].values.astype(np.float32).reshape(-1, 1)

        # Create model
        self.model = create_model(
            model_type=model_config['type'],
            input_dim=len(self.feature_cols),
            **model_config.get('kwargs', {})
        )

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Loss function
        self.criterion = nn.BCELoss()

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def _create_dataloader(self) -> DataLoader:
        """Create PyTorch dataloader for local data."""
        dataset = TensorDataset(
            torch.FloatTensor(self.X),
            torch.FloatTensor(self.y)
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def local_train(self, global_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train local model and return update.

        Args:
            global_weights: Optional global model weights to start from

        Returns:
            Dict containing:
                - 'update': Flattened model update (weights or gradients)
                - 'n_samples': Number of local training samples
                - 'metrics': Training metrics (loss, accuracy)
                - 'weights': Full model weights after training
        """
        # Load global weights if provided
        if global_weights is not None:
            self.model.set_weights(global_weights)

        # Store initial weights for gradient computation
        initial_weights = self.model.get_weights().copy()

        # Train locally
        train_loss, train_acc = self._train_loop()

        # Get update
        final_weights = self.model.get_weights()
        weight_update = final_weights - initial_weights

        # Compute metrics
        metrics = {
            'loss': float(train_loss),
            'accuracy': float(train_acc),
            'n_samples': len(self.dataset),
            'n_epochs': self.local_epochs
        }

        return {
            'update': weight_update,
            'weights': final_weights,
            'client_id': self.client_id,
            'metrics': metrics
        }

    def _train_loop(self) -> Tuple[float, float]:
        """Execute local training loop.

        Returns:
            (final_loss, final_accuracy) tuple
        """
        self.model.train()
        dataloader = self._create_dataloader()

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                # Forward pass
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Statistics
                epoch_loss += loss.item() * len(batch_X)
                predicted = (predictions > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += len(batch_X)

            epoch_loss /= total
            epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def evaluate(self, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate model on local data.

        Args:
            weights: Optional model weights to evaluate

        Returns:
            Dict with evaluation metrics
        """
        if weights is not None:
            self.model.set_weights(weights)

        self.model.eval()
        dataloader = self._create_dataloader()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)

                total_loss += loss.item() * len(batch_X)
                predicted = (predictions > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += len(batch_X)

        metrics = {
            'loss': total_loss / total,
            'accuracy': correct / total,
            'n_samples': total
        }

        return metrics

    def get_dataset_properties(self) -> Dict[str, Any]:
        """Get properties of local dataset.

        Returns:
            Dict with dataset statistics
        """
        from ..attacks.property_extractor import extract_all_properties

        properties = extract_all_properties(self.dataset)
        properties['client_id'] = self.client_id
        properties['feature_cols'] = self.feature_cols

        return properties

    def get_update_size(self) -> int:
        """Return size of model update in bytes."""
        return self.model.get_weights().nbytes


class DPEnabledClient(FederatedClient):
    """Client with differential privacy support."""

    def __init__(
        self,
        client_id: int,
        dataset: pd.DataFrame,
        model_config: Dict[str, Any],
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer_type: str = 'adam',
        dp_enabled: bool = False,
        dp_noise_multiplier: float = 1.0,
        dp_max_grad_norm: float = 1.0
    ):
        """Initialize DP-enabled client.

        Args:
            client_id: Unique client identifier
            dataset: Local training data
            model_config: Configuration for model creation
            local_epochs: Number of local training epochs
            batch_size: Batch size for local training
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer
            dp_enabled: Whether to enable differential privacy
            dp_noise_multiplier: Noise multiplier for DP
            dp_max_grad_norm: Maximum gradient norm for clipping
        """
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            model_config=model_config,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type
        )

        self.dp_enabled = dp_enabled
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_max_grad_norm = dp_max_grad_norm

    def _apply_dp(self) -> None:
        """Apply differential privacy to gradients."""
        if not self.dp_enabled:
            return

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.dp_max_grad_norm
        )

        # Add noise
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.dp_noise_multiplier * self.dp_max_grad_norm
                    param.grad += noise

    def _train_loop(self) -> Tuple[float, float]:
        """Execute local training loop with DP.

        Returns:
            (final_loss, final_accuracy) tuple
        """
        self.model.train()
        dataloader = self._create_dataloader()

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                # Forward pass
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Apply DP
                self._apply_dp()

                self.optimizer.step()

                # Statistics
                epoch_loss += loss.item() * len(batch_X)
                predicted = (predictions > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += len(batch_X)

            epoch_loss /= total
            epoch_acc = correct / total

        return epoch_loss, epoch_acc
