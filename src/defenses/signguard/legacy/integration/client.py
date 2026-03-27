"""
SignGuard Client for Flower Framework

Implements client-side signing and model update submission.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)

from ..crypto.key_manager import KeyManager
from ..crypto.signature_handler import SignatureHandler, SignedUpdate


class SignGuardClient(fl.client.NumPyClient):
    """
    Flower client with SignGuard cryptographic signing.

    Features:
    - ECDSA signature on model updates
    - Public key registration with server
    - Standard federated learning training
    """

    def __init__(self,
                 client_id: str,
                 model,
                 train_loader,
                 test_loader,
                 device: str = "cpu",
                 key_dir: Optional[str] = None,
                 existing_private_key: Optional[bytes] = None):
        """
        Initialize SignGuardClient.

        Args:
            client_id: Unique client identifier
            model: PyTorch model
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device for training (cpu/cuda)
            key_dir: Directory for key storage (generates new if None and no existing key)
            existing_private_key: Existing private key in PEM format
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        # Cryptographic components
        self.key_manager = KeyManager(key_dir=key_dir)

        # Use existing key or generate new one
        if existing_private_key is not None:
            # Load existing key
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend

            private_key = serialization.load_pem_private_key(
                existing_private_key,
                password=None,
                backend=default_backend()
            )
            # For simplicity, we still register in key_manager
            # In production, you'd properly store the key
        elif client_id not in self.key_manager.list_clients():
            # Generate new key pair
            self.private_key_pem, self.public_key_pem = \
                self.key_manager.generate_key_pair(client_id)
        else:
            # Use existing key
            self.private_key_pem = None  # Will be loaded when needed
            self.public_key_pem = self.key_manager.get_public_key_pem(client_id)

        self.signature_handler = SignatureHandler()

        # Training parameters (set by server)
        self.local_epochs = 5
        self.batch_size = 32
        self.learning_rate = 0.01

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get current model parameters.

        Args:
            config: Configuration from server

        Returns:
            Model parameters as list of numpy arrays
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters.

        Args:
            parameters: Model parameters as list of numpy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally and return signed update.

        Args:
            parameters: Global model parameters from server
            config: Training configuration (epochs, batch_size, etc.)

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Parse config
        self.local_epochs = config.get('local_epochs', self.local_epochs)
        round_num = config.get('server_round', 0)

        # Set global parameters
        self.set_parameters(parameters)

        # Train locally
        num_examples = self._train()

        # Get updated parameters
        updated_params = self.get_parameters(config={})

        # Sign the update
        signature = self._sign_update(updated_params, round_num)

        # Return with signature in metrics
        metrics = {
            'client_id': self.client_id,
            'signature': signature.hex(),  # Convert to hex for transmission
            'timestamp': time.time(),
            'num_examples': num_examples
        }

        return updated_params, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on test set.

        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set parameters
        self.set_parameters(parameters)

        # Evaluate
        loss, accuracy = self._evaluate()

        # Return metrics
        metrics = {
            'accuracy': accuracy,
            'client_id': self.client_id
        }

        return loss, len(self.test_loader.dataset), metrics

    def get_public_key(self) -> bytes:
        """Get client's public key in PEM format."""
        return self.public_key_pem

    def _sign_update(self, update: List[np.ndarray], round_num: int) -> bytes:
        """
        Sign model update with ECDSA.

        Args:
            update: Model parameters
            round_num: Current FL round

        Returns:
            DER-encoded signature
        """
        # Load private key
        private_key = self.key_manager.get_private_key(self.client_id)

        # Sign
        timestamp = time.time()
        signature = self.signature_handler.sign_parameters(
            private_key=private_key,
            parameters=update,
            round_num=round_num,
            timestamp=timestamp
        )

        return signature

    def _train(self) -> int:
        """
        Train model for one epoch.

        Returns:
            Number of training examples
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Setup
        self.model.train()
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # Training loop
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)

                # Backward pass
                loss.backward()
                optimizer.step()

        return len(self.train_loader.dataset)

    def _evaluate(self) -> Tuple[float, float]:
        """
        Evaluate model on test set.

        Returns:
            Tuple of (loss, accuracy)
        """
        import torch
        import torch.nn as nn

        self.model.eval()
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                test_loss += criterion(output, target).item()

                # Accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = correct / total if total > 0 else 0.0

        return test_loss, accuracy


class SignGuardClientProxy:
    """
    Proxy for SignGuard client registration and key exchange.

    Used during initial setup where clients register their public keys.
    """

    def __init__(self, client_id: str, public_key_pem: bytes):
        """
        Initialize client proxy.

        Args:
            client_id: Client identifier
            public_key_pem: Client's public key in PEM format
        """
        self.client_id = client_id
        self.public_key_pem = public_key_pem

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'client_id': self.client_id,
            'public_key': self.public_key_pem.decode('utf-8', errors='ignore')
        }


def create_client(client_id: str,
                  model,
                  train_loader,
                  test_loader,
                  device: str = "cpu",
                  key_dir: Optional[str] = None) -> SignGuardClient:
    """
    Factory function to create SignGuard client.

    Args:
        client_id: Client identifier
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device for training
        key_dir: Directory for key storage

    Returns:
        SignGuardClient instance
    """
    return SignGuardClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        key_dir=key_dir
    )
