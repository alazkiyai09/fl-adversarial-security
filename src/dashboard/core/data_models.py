"""
Core data models for FL Security Dashboard.
Uses Pydantic for type safety, validation, and serialization.
"""

from datetime import datetime
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import numpy as np


class ClientMetric(BaseModel):
    """Metrics for a single client in a training round."""
    client_id: int
    accuracy: float = Field(ge=0, le=1, description="Client accuracy")
    loss: float = Field(ge=0, description="Client loss")
    data_size: int = Field(gt=0, description="Number of training samples")
    training_time: float = Field(ge=0, description="Training time in seconds")
    status: Literal["active", "idle", "dropped", "anomaly"] = "active"

    # Anomaly detection fields
    anomaly_score: float = Field(default=0.0, ge=0, le=1, description="Anomaly probability")
    update_norm: float = Field(default=0.0, ge=0, description="L2 norm of model update")
    reputation_score: float = Field(default=1.0, ge=0, le=1, description="Client reputation")

    @field_validator("update_norm")
    @classmethod
    def validate_update_norm(cls, v: float) -> float:
        """Ensure update norm is reasonable."""
        if v > 1000:  # Prevent extreme values
            return 1000.0
        return v


class TrainingRound(BaseModel):
    """Complete data for one training round."""
    round_num: int = Field(ge=0, description="Round number")
    timestamp: datetime = Field(default_factory=datetime.now)
    global_loss: float = Field(ge=0, description="Global training loss")
    global_accuracy: float = Field(ge=0, le=1, description="Global accuracy")
    per_client_metrics: List[ClientMetric] = Field(default_factory=list)

    # Convergence metrics
    loss_delta: float = Field(default=0.0, description="Change from previous round")
    accuracy_delta: float = Field(default=0.0, description="Change from previous round")

    # Security events in this round
    security_events: List["SecurityEvent"] = Field(default_factory=list)

    # Privacy tracking
    epsilon_spent: float = Field(default=0.0, ge=0, description="DP epsilon spent this round")

    @field_validator("per_client_metrics")
    @classmethod
    def validate_clients(cls, v: List[ClientMetric]) -> List[ClientMetric]:
        """Ensure unique client IDs."""
        client_ids = [m.client_id for m in v]
        if len(client_ids) != len(set(client_ids)):
            raise ValueError("Duplicate client IDs detected")
        return v


class SecurityEvent(BaseModel):
    """Security-related event (attack detected, defense activated, etc.)."""
    event_id: str = Field(description="Unique event identifier")
    event_type: Literal[
        "attack_detected",
        "attack_successful",
        "defense_activated",
        "client_flagged",
        "privacy_violation",
        "anomaly_detected"
    ] = Field(description="Type of security event")

    severity: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Event severity"
    )

    message: str = Field(description="Human-readable event description")
    round_num: int = Field(ge=0, description="Round when event occurred")

    # Attack-specific fields
    attack_type: Optional[Literal[
        "label_flipping",
        "backdoor",
        "byzantine",
        "poisoning",
        "gradient_leakage"
    ]] = Field(default=None, description="Attack type if applicable")

    affected_clients: List[int] = Field(default_factory=list, description="Client IDs involved")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Detection confidence")

    timestamp: datetime = Field(default_factory=datetime.now)
    resolved: bool = Field(default=False, description="Whether event was resolved")


class PrivacyBudget(BaseModel):
    """Differential privacy budget tracking."""
    epsilon_total: float = Field(gt=0, description="Total privacy budget")
    epsilon_spent: float = Field(default=0.0, ge=0, description="Epsilon used so far")
    epsilon_remaining: float = Field(ge=0, description="Remaining epsilon")

    delta: float = Field(default=1e-5, ge=0, description="Delta parameter")
    per_round_epsilon: List[float] = Field(default_factory=list, description="Epsilon per round")

    # Secure aggregation status
    secure_aggregation: bool = Field(default=True, description="Using secure aggregation")
    encryption_method: Optional[Literal["Paillier", "ElGamal", "BGW"]] = Field(
        default="Paillier",
        description="Encryption method used"
    )

    @field_validator("epsilon_remaining")
    @classmethod
    def validate_remaining(cls, v: float, info) -> float:
        """Ensure remaining doesn't exceed total."""
        if "epsilon_total" in info.data and v > info.data["epsilon_total"]:
            return info.data["epsilon_total"]
        return v


class ExperimentResult(BaseModel):
    """Results from a complete experiment run."""
    experiment_id: str = Field(description="Unique experiment identifier")
    name: str = Field(description="Experiment name")
    config: "FLConfig" = Field(description="FL configuration used")

    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(default=None, description="Experiment completion time")

    # Training history
    training_history: List[TrainingRound] = Field(default_factory=list)

    # Final metrics
    final_accuracy: float = Field(ge=0, le=1, description="Final model accuracy")
    final_loss: float = Field(ge=0, description="Final model loss")

    # Security outcomes
    total_attacks_detected: int = Field(default=0, ge=0)
    total_defense_activations: int = Field(default=0, ge=0)
    attack_success_rate: float = Field(default=0.0, ge=0, le=1)

    # Privacy outcomes
    final_epsilon_spent: float = Field(default=0.0, ge=0)

    # Performance
    total_time_seconds: float = Field(default=0.0, ge=0)
    rounds_completed: int = Field(default=0, ge=0)

    status: Literal["running", "completed", "failed", "stopped"] = Field(default="running")


class FLConfig(BaseModel):
    """Federated Learning configuration."""
    # Training parameters
    num_rounds: int = Field(default=100, ge=1, description="Total training rounds")
    num_clients: int = Field(default=10, ge=1, description="Number of clients")
    clients_per_round: int = Field(default=10, ge=1, description="Clients selected per round")

    # Model parameters
    learning_rate: float = Field(default=0.01, gt=0, description="Server learning rate")
    batch_size: int = Field(default=32, ge=1, description="Local batch size")
    local_epochs: int = Field(default=5, ge=1, description="Local training epochs")

    # Data distribution
    data_distribution: Literal["iid", "non_iid_dirichlet", "non_iid_pathological"] = Field(
        default="non_iid_dirichlet",
        description="Data distribution across clients"
    )
    dirichlet_alpha: float = Field(default=0.5, gt=0, description="Dirichlet concentration")

    # Privacy parameters
    use_dp: bool = Field(default=False, description="Use differential privacy")
    dp_epsilon: float = Field(default=1.0, gt=0, description="DP epsilon per round")
    dp_delta: float = Field(default=1e-5, gt=0, description="DP delta")
    noise_multiplier: float = Field(default=0.1, ge=0, description="DP noise multiplier")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Gradient clipping norm")

    # Defense parameters
    defense_type: Literal["none", "signguard", "krum", "foolsgold", "trim_mean"] = Field(
        default="signguard",
        description="Defense mechanism"
    )

    # Simulation parameters
    drop_rate: float = Field(default=0.0, ge=0, le=1, description="Client drop rate")
    seed: int = Field(default=42, description="Random seed")


class AttackConfig(BaseModel):
    """Attack configuration for simulation."""
    attack_type: Literal[
        "label_flipping",
        "backdoor",
        "byzantine",
        "poisoning",
        "gradient_leakage"
    ] = Field(description="Type of attack")

    # Attack timing
    start_round: int = Field(default=10, ge=0, description="Round to start attack")
    end_round: Optional[int] = Field(default=None, description="Round to end attack (None = until end)")

    # Attacker configuration
    num_attackers: int = Field(default=1, ge=1, description="Number of malicious clients")
    attacker_ids: List[int] = Field(default_factory=list, description="Specific attacker client IDs")

    # Attack-specific parameters
    label_flip_ratio: float = Field(default=1.0, ge=0, le=1, description="Fraction of labels to flip")
    backdoor_target_class: int = Field(default=0, ge=0, description="Target class for backdoor")
    backdoor_trigger_pattern: float = Field(default=0.0, description="Pattern to inject")
    byzantine_type: Literal["random", "sign_flip", "scaled"] = Field(
        default="sign_flip",
        description="Byzantine attack variant"
    )
    poison_magnitude: float = Field(default=10.0, gt=0, description="Magnitude of poisoning")

    # Expected effectiveness
    expected_impact: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Expected attack impact"
    )


class DefenseConfig(BaseModel):
    """Defense configuration."""
    defense_type: Literal["signguard", "krum", "foolsgold", "trim_mean", "median"] = Field(
        description="Defense mechanism"
    )

    # Detection thresholds
    anomaly_threshold: float = Field(default=0.5, ge=0, le=1, description="Anomaly detection threshold")
    reputation_threshold: float = Field(default=0.3, ge=0, le=1, description="Reputation cutoff")

    # Krum-specific
    krum_num_attackers: int = Field(default=2, ge=0, description="Expected number of attackers")

    # Trim-mean specific
    trim_ratio: float = Field(default=0.2, ge=0, le=0.5, description="Fraction to trim from each end")

    # SignGuard-specific
    signguard_window_size: int = Field(default=5, ge=1, description="Reputation history window")
    signguard_decay_factor: float = Field(default=0.9, ge=0, le=1, description="Reputation decay")

    # Action on detection
    action_on_detection: Literal["drop", "downweight", "flag_only"] = Field(
        default="downweight",
        description="Action when attack detected"
    )


# Forward references for circular dependencies
TrainingRound.model_rebuild()
SecurityEvent.model_rebuild()
ExperimentResult.model_rebuild()


class ClientUpdateVector(BaseModel):
    """Raw model update vector from a client."""
    client_id: int
    round_num: int
    update_vector: List[float] = Field(description="Flattened model update")
    update_shape: tuple = Field(description="Original tensor shape")

    class Config:
        arbitrary_types_allowed = True


class ClusterVisualization(BaseModel):
    """Data for client clustering visualization."""
    client_ids: List[int]
    tsne_coordinates: List[tuple] = Field(description="2D coordinates from t-SNE")
    cluster_labels: List[int] = Field(description="Cluster assignment for each client")
    anomaly_flags: List[bool] = Field(description="Whether each client is flagged")


class AggregationResult(BaseModel):
    """Result of one aggregation round."""
    round_num: int
    aggregated_update: List[float]
    clients_included: List[int]
    clients_excluded: List[int] = Field(default_factory=list)
    exclusion_reasons: Dict[int, str] = Field(default_factory=dict)

    # Defense metrics
    defense_applied: bool = Field(default=False)
    defense_type: Optional[str] = Field(default=None)
    clients_downweighted: Dict[int, float] = Field(default_factory=dict)
