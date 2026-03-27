"""Core type definitions for SignGuard."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import torch
import time


@dataclass
class ModelUpdate:
    """Unsigned model update from a client.

    Attributes:
        client_id: Unique identifier for the client
        round_num: Federated learning round number
        parameters: Model parameters as a dictionary of layer names to tensors
        num_samples: Number of training samples used
        metrics: Training metrics (loss, accuracy, etc.)
        timestamp: Unix timestamp of update creation
    """

    client_id: str
    round_num: int
    parameters: Dict[str, torch.Tensor]
    num_samples: int
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary with all data in serializable format
        """
        return {
            "client_id": self.client_id,
            "round_num": self.round_num,
            "parameters": {
                name: param.cpu().numpy().tolist()
                for name, param in self.parameters.items()
            },
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_serializable(
        cls, data: Dict[str, Any], device: str = "cpu"
    ) -> "ModelUpdate":
        """Create from JSON-serializable dictionary.

        Args:
            data: Serialized dictionary
            device: Torch device to place tensors on

        Returns:
            ModelUpdate instance
        """
        parameters = {
            name: torch.tensor(param, device=device)
            for name, param in data["parameters"].items()
        }
        return cls(
            client_id=data["client_id"],
            round_num=data["round_num"],
            parameters=parameters,
            num_samples=data["num_samples"],
            metrics=data.get("metrics", {}),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class SignedUpdate:
    """Cryptographically signed model update.

    Attributes:
        update: The model update being signed
        signature: Base64-encoded ECDSA signature
        public_key: Base64-encoded public key
        algorithm: Signature algorithm (default: ECDSA)
        nonce: Server-provided nonce for replay protection
    """

    update: ModelUpdate
    signature: str
    public_key: str
    algorithm: str = "ECDSA"
    nonce: Optional[str] = None

    def is_valid_signature(self) -> bool:
        """Check if signature format is valid.

        Returns:
            True if signature is properly formatted
        """
        return (
            isinstance(self.signature, str)
            and len(self.signature) > 0
            and isinstance(self.public_key, str)
            and len(self.public_key) > 0
        )


@dataclass
class AnomalyScore:
    """Multi-factor anomaly detection result.

    Attributes:
        magnitude_score: L2 norm anomaly score [0, 1]
        direction_score: Cosine similarity anomaly score [0, 1]
        loss_score: Loss deviation anomaly score [0, 1]
        combined_score: Weighted combination of all scores [0, 1]
    """

    magnitude_score: float
    direction_score: float
    loss_score: float
    combined_score: float

    def __post_init__(self):
        """Validate scores are in valid range."""
        for score_name, score_value in [
            ("magnitude", self.magnitude_score),
            ("direction", self.direction_score),
            ("loss", self.loss_score),
            ("combined", self.combined_score),
        ]:
            if not 0.0 <= score_value <= 1.0:
                raise ValueError(
                    f"{score_name}_score must be in [0, 1], got {score_value}"
                )

    def is_anomalous(self, threshold: float = 0.7) -> bool:
        """Check if combined score exceeds threshold.

        Args:
            threshold: Anomaly detection threshold

        Returns:
            True if update is considered anomalous
        """
        return self.combined_score > threshold


@dataclass
class ReputationInfo:
    """Client reputation information.

    Attributes:
        client_id: Unique client identifier
        reputation: Current reputation score [0, 1]
        num_contributions: Number of valid contributions made
        last_update_round: Last round this client contributed
        detection_history: Recent anomaly scores (deque-like)
    """

    client_id: str
    reputation: float
    num_contributions: int
    last_update_round: int
    detection_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Validate reputation is in valid range."""
        if not 0.0 <= self.reputation <= 1.0:
            raise ValueError(f"Reputation must be in [0, 1], got {self.reputation}")

    def add_detection_score(self, score: float) -> None:
        """Add a new anomaly score to history.

        Args:
            score: Anomaly score to add
        """
        self.detection_history.append(score)

    def get_average_anomaly(self) -> float:
        """Get average anomaly score from history.

        Returns:
            Average anomaly score, or 0 if no history
        """
        if not self.detection_history:
            return 0.0
        return sum(self.detection_history) / len(self.detection_history)


@dataclass
class AggregationResult:
    """Result of a single aggregation round.

    Attributes:
        global_model: Aggregated global model parameters
        participating_clients: Client IDs that participated
        excluded_clients: Client IDs excluded (failed signature or high anomaly)
        reputation_updates: Client ID -> new reputation mapping
        round_num: Round number
        execution_time: Time taken for aggregation (seconds)
        metadata: Additional metadata
    """

    global_model: Dict[str, torch.Tensor]
    participating_clients: List[str]
    excluded_clients: List[str]
    reputation_updates: Dict[str, float]
    round_num: int
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_exclusion_reasons(self) -> Dict[str, str]:
        """Get reasons for client exclusions from metadata.

        Returns:
            Dict mapping client_id -> exclusion reason
        """
        return self.metadata.get("exclusion_reasons", {})


@dataclass
class ClientConfig:
    """Configuration for a SignGuard client.

    Attributes:
        client_id: Unique client identifier
        local_epochs: Number of local training epochs
        learning_rate: Learning rate for local training
        batch_size: Batch size for training
        device: Device to train on
        optimizer: Optimizer type (sgd, adam)
    """

    client_id: str
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    device: str = "cpu"
    optimizer: str = "sgd"


@dataclass
class ServerConfig:
    """Configuration for SignGuard server.

    Attributes:
        num_rounds: Total number of training rounds
        num_clients_per_round: Clients selected per round
        min_clients_required: Minimum clients for aggregation
        anomaly_threshold: Threshold for anomaly detection
        min_reputation_threshold: Minimum reputation to participate
    """

    num_rounds: int = 100
    num_clients_per_round: int = 10
    min_clients_required: int = 5
    anomaly_threshold: float = 0.7
    min_reputation_threshold: float = 0.1


@dataclass
class ExperimentConfig:
    """Configuration for experiments.

    Attributes:
        seed: Random seed for reproducibility
        dataset: Dataset name
        num_clients: Total number of clients
        num_byzantine: Number of Byzantine (malicious) clients
        attack_type: Type of attack to simulate
        defense_type: Type of defense to use
        output_dir: Directory to save results
    """

    seed: int = 42
    dataset: str = "credit_card"
    num_clients: int = 20
    num_byzantine: int = 4
    attack_type: str = "label_flip"
    defense_type: str = "signguard"
    output_dir: str = "results"
