"""SignGuard client with cryptographic signing."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from cryptography.hazmat.primitives.asymmetric import ec
from tqdm import tqdm

from src.defenses.signguard_full.legacy.core.types import ModelUpdate, SignedUpdate, ClientConfig
from src.defenses.signguard_full.legacy.crypto.signature import SignatureManager
from src.defenses.signguard_full.legacy.utils.serialization import serialize_model


class SignGuardClient:
    """SignGuard client with cryptographic signing.

    Trains local model and creates signed updates for federated learning.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        signature_manager: SignatureManager,
        private_key: ec.EllipticCurvePrivateKey,
        config: Optional[ClientConfig] = None,
    ):
        """Initialize SignGuard client.

        Args:
            client_id: Unique client identifier
            model: PyTorch model to train
            train_loader: Training data loader
            signature_manager: Signature manager for signing
            private_key: Client's private key
            config: Client configuration (uses defaults if None)
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.signature_manager = signature_manager
        self.private_key = private_key
        
        # Use provided config or create default
        self.config = config or ClientConfig(client_id=client_id)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Device
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config.

        Returns:
            Optimizer instance
        """
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.optimizer_momentum if hasattr(self.config, 'optimizer_momentum') else 0.9,
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def train(
        self,
        global_model: Dict[str, torch.Tensor],
        current_round: int = 0,
        verbose: bool = False,
    ) -> ModelUpdate:
        """Train locally on global model.

        Args:
            global_model: Current global model parameters
            current_round: Current FL round number
            verbose: Whether to show progress bar

        Returns:
            ModelUpdate with trained parameters and metrics
        """
        # Load global model
        self.model.load_state_dict(global_model)
        self.model.train()
        
        # Store initial parameters for computing update
        initial_params = {
            name: param.clone().detach()
            for name, param in self.model.state_dict().items()
        }
        
        # Local training
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        epoch_iter = range(self.config.local_epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc=f"Client {self.client_id}")

        for epoch in epoch_iter:
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Compute loss
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += inputs.size(0)
        
        # Compute metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Compute parameter update (difference)
        trained_params = self.model.state_dict()
        update_params = {
            name: trained_params[name] - initial_params[name]
            for name in trained_params.keys()
        }
        
        # Create model update
        model_update = ModelUpdate(
            client_id=self.client_id,
            round_num=current_round,
            parameters=update_params,
            num_samples=total_samples,
            metrics={
                "loss": avg_loss,
                "accuracy": accuracy,
            },
        )
        
        return model_update

    def sign_update(self, update: ModelUpdate) -> SignedUpdate:
        """Create signed update.

        Args:
            update: Unsigned model update

        Returns:
            SignedUpdate with ECDSA signature
        """
        # Get public key
        public_key = self.private_key.public_key()
        public_key_str = self.signature_manager.serialize_public_key(public_key)
        
        # Sign the update
        signature = self.signature_manager.sign_update(update, self.private_key)
        
        # Create signed update
        signed_update = SignedUpdate(
            update=update,
            signature=signature,
            public_key=public_key_str,
            algorithm="ECDSA",
        )
        
        return signed_update

    def get_public_key(self) -> str:
        """Get serialized public key.

        Returns:
            Base64-encoded public key string
        """
        public_key = self.private_key.public_key()
        return self.signature_manager.serialize_public_key(public_key)

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk.

        Args:
            filepath: Path to save model
        """
        serialize_model(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """Load model from disk.

        Args:
            filepath: Path to load model from
        """
        from src.defenses.signguard_full.legacy.utils.serialization import deserialize_model
        deserialize_model(filepath, self.model, self.device)

    def evaluate(
        self,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            test_loader: Test data loader (uses train_loader if None)

        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        data_loader = test_loader or self.train_loader
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += inputs.size(0)
        
        return {
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        }


def create_client(
    client_id: str,
    model: nn.Module,
    train_loader: DataLoader,
    signature_manager: SignatureManager,
    private_key: ec.EllipticCurvePrivateKey,
    **config_kwargs,
) -> SignGuardClient:
    """Helper function to create a SignGuard client.

    Args:
        client_id: Client identifier
        model: PyTorch model
        train_loader: Training data
        signature_manager: Signature manager
        private_key: Private key
        **config_kwargs: Additional config arguments

    Returns:
        Configured SignGuardClient
    """
    config = ClientConfig(client_id=client_id, **config_kwargs)
    return SignGuardClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        signature_manager=signature_manager,
        private_key=private_key,
        config=config,
    )
