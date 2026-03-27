"""Core components for SignGuard."""

from src.defenses.signguard_full.legacy.core.types import (
    ModelUpdate,
    SignedUpdate,
    AnomalyScore,
    ReputationInfo,
    AggregationResult,
    ClientConfig,
    ServerConfig,
    ExperimentConfig,
)
from src.defenses.signguard_full.legacy.core.client import SignGuardClient, create_client
from src.defenses.signguard_full.legacy.core.server import SignGuardServer

__all__ = [
    # Types
    "ModelUpdate",
    "SignedUpdate",
    "AnomalyScore",
    "ReputationInfo",
    "AggregationResult",
    "ClientConfig",
    "ServerConfig",
    "ExperimentConfig",
    # Main components
    "SignGuardClient",
    "SignGuardServer",
    "create_client",
]
