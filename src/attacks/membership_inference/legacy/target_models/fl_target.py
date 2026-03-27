"""
Target Federated Learning Model

This module trains the target FL model that will be attacked.
The target model is trained BEFORE any attack code is executed,
ensuring no information leakage from attack to target.

IMPORTANT: This file should be run independently from attack experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import numpy as np
import pickle
import os


class FraudDetectionNN(nn.Module):
    """
    Neural network for fraud detection.

    Architecture: Simple feedforward network suitable for tabular fraud data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize fraud detection model.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            num_classes: Number of output classes (binary: 2)
            dropout: Dropout rate for regularization
        """
        super(FraudDetectionNN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class FLTargetTrainer:
    """
    Trains target FL model using Federated Averaging (FedAvg).

    This simulates the FL training process that an attacker wants to infer
    membership from.
    """

    def __init__(
        self,
        model: nn.Module,
        n_clients: int = 10,
        local_epochs: int = 5,
        client_lr: float = 0.01,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize FL target trainer.

        Args:
            model: PyTorch model to train
            n_clients: Number of FL clients
            local_epochs: Local training epochs per client
            client_lr: Learning rate for local training
            batch_size: Batch size for local training
            device: Device to train on
        """
        self.model = model.to(device)
        self.n_clients = n_clients
        self.local_epochs = local_epochs
        self.client_lr = client_lr
        self.batch_size = batch_size
        self.device = device

        # Store model history for temporal analysis
        self.model_history = []  # Global model after each round

    def train_client_local(
        self,
        client_model: nn.Module,
        client_data: DataLoader,
        criterion: nn.Module
    ) -> nn.Module:
        """
        Train client model locally on client data.

        Args:
            client_model: Client's copy of global model
            client_data: Client's local training data
            criterion: Loss function

        Returns:
            Updated client model
        """
        client_model.train()
        optimizer = optim.SGD(
            client_model.parameters(),
            lr=self.client_lr,
            momentum=0.9
        )

        for epoch in range(self.local_epochs):
            for x_batch, y_batch in client_data:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = client_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        return client_model

    def federated_averaging(
        self,
        client_models: List[nn.Module],
        client_sizes: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client models using FedAvg.

        Args:
            client_models: List of client model state dicts
            client_sizes: Number of samples per client

        Returns:
            Averaged global model state dict
        """
        global_state = client_models[0].state_dict()
        aggregated_state = {}

        total_samples = sum(client_sizes)

        for key in global_state.keys():
            # Weighted average of client parameters
            aggregated_state[key] = sum(
                client_model.state_dict()[key] * client_size
                for client_model, client_size in zip(client_models, client_sizes)
            ) / total_samples

        return aggregated_state

    def train_fl_model(
        self,
        client_datasets: List[DataLoader],
        n_rounds: int = 20,
        verbose: bool = True
    ) -> nn.Module:
        """
        Train target FL model over multiple communication rounds.

        Args:
            client_datasets: List of client training datasets
            n_rounds: Number of FL communication rounds
            verbose: Print training progress

        Returns:
            Trained global model
        """
        criterion = nn.CrossEntropyLoss()

        for round_idx in range(n_rounds):
            if verbose:
                print(f"Round {round_idx + 1}/{n_rounds}")

            # Initialize client models from global model
            client_models = [
                type(self.model)(*(self.model.__dict__.values())).to(self.device)
                for _ in range(self.n_clients)
            ]

            # Load global weights into each client model
            global_state = self.model.state_dict()
            for client_model in client_models:
                client_model.load_state_dict(global_state)

            # Train each client locally
            trained_clients = []
            client_sizes = []

            for i, (client_model, client_data) in enumerate(zip(client_models, client_datasets)):
                client_model = self.train_client_local(client_model, client_data, criterion)
                trained_clients.append(client_model)
                client_sizes.append(len(client_data.dataset))

            # Aggregate client models
            aggregated_state = self.federated_averaging(trained_clients, client_sizes)

            # Update global model
            self.model.load_state_dict(aggregated_state)

            # Save model state for temporal analysis
            self.model_history.append({
                'round': round_idx,
                'state_dict': {k: v.cpu().clone() for k, v in aggregated_state.items()}
            })

            if verbose:
                # Compute training loss
                self.model.eval()
                total_loss = 0
                total_samples = 0

                with torch.no_grad():
                    for client_data in client_datasets:
                        for x, y in client_data:
                            x, y = x.to(self.device), y.to(self.device)
                            outputs = self.model(x)
                            loss = criterion(outputs, y)
                            total_loss += loss.item() * len(x)
                            total_samples += len(x)

                avg_loss = total_loss / total_samples
                print(f"  Avg training loss: {avg_loss:.4f}")

        return self.model

    def save_target_model(
        self,
        save_path: str
    ):
        """
        Save trained target model.

        Args:
            save_path: Path to save model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_history': self.model_history,
            'config': {
                'n_clients': self.n_clients,
                'local_epochs': self.local_epochs,
                'client_lr': self.client_lr
            }
        }, save_path)

        print(f"✓ Target model saved to {save_path}")

    def load_target_model(
        self,
        load_path: str
    ):
        """
        Load trained target model.

        Args:
            load_path: Path to load model from
        """
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_history = checkpoint['model_history']

        print(f"✓ Target model loaded from {load_path}")


def create_client_splits(
    full_dataset,
    n_clients: int = 10,
    samples_per_client: int = None,
    batch_size: int = 32,
    random_seed: int = 42
) -> List[DataLoader]:
    """
    Create client datasets for FL simulation.

    Args:
        full_dataset: Full training dataset
        n_clients: Number of clients
        samples_per_client: Samples per client (default: balanced split)
        batch_size: Batch size for DataLoader
        random_seed: Random seed for reproducibility

    Returns:
        List of client DataLoaders
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    n_total = len(full_dataset)

    if samples_per_client is None:
        samples_per_client = n_total // n_clients

    indices = np.random.permutation(n_total)

    client_loaders = []

    for i in range(n_clients):
        start_idx = i * samples_per_client
        end_idx = min(start_idx + samples_per_client, n_total)

        client_indices = indices[start_idx:end_idx]

        from torch.utils.data import Subset
        client_dataset = Subset(full_dataset, client_indices)
        client_loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        client_loaders.append(client_loader)

    return client_loaders


if __name__ == "__main__":
    """
    Example usage: Train target FL model independently.

    Run this BEFORE running any attack experiments.
    """
    # Load data (example: using synthetic data for demonstration)
    # In practice, load your actual fraud detection dataset
    n_samples = 10000
    n_features = 20

    # Create synthetic data
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))

    from torch.utils.data import TensorDataset
    dataset = TensorDataset(X, y)

    # Create client splits
    client_datasets = create_client_splits(
        dataset,
        n_clients=10,
        batch_size=32
    )

    # Initialize model
    model = FraudDetectionNN(
        input_dim=n_features,
        hidden_dims=[128, 64, 32],
        num_classes=2
    )

    # Train FL model
    trainer = FLTargetTrainer(
        model=model,
        n_clients=10,
        local_epochs=5,
        client_lr=0.01,
        device='cpu'
    )

    print("Training target FL model...")
    trained_model = trainer.train_fl_model(
        client_datasets=client_datasets,
        n_rounds=20,
        verbose=True
    )

    # Save target model
    save_path = "data/processed/target_fl_model.pt"
    trainer.save_target_model(save_path)

    print("\n✓ Target model training complete.")
    print("⚠️  IMPORTANT: Run attack experiments on a SEPARATE process/run")
    print("   to ensure no information leakage from target to attack.")
