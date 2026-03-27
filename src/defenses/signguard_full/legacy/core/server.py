"""SignGuard server with verification and anomaly detection."""

import torch
import torch.nn as nn
import time
from typing import List, Dict, Optional, Set
from tqdm import tqdm

from src.defenses.signguard_full.legacy.core.types import (
    ModelUpdate,
    SignedUpdate,
    AnomalyScore,
    AggregationResult,
    ServerConfig,
)
from src.defenses.signguard_full.legacy.crypto.signature import SignatureManager
from src.defenses.signguard_full.legacy.detection.ensemble import EnsembleDetector
from src.defenses.signguard_full.legacy.reputation.decay_reputation import DecayReputationSystem
from src.defenses.signguard_full.legacy.aggregation.weighted_aggregator import WeightedAggregator
from src.defenses.signguard_full.legacy.utils.metrics import compute_accuracy


class SignGuardServer:
    """SignGuard server with verification, detection, and reputation.

    Orchestrates the federated learning aggregation pipeline with
    signature verification, anomaly detection, reputation scoring,
    and weighted aggregation.
    """

    def __init__(
        self,
        global_model: nn.Module,
        signature_manager: SignatureManager,
        detector: Optional[EnsembleDetector] = None,
        reputation_system: Optional[DecayReputationSystem] = None,
        aggregator: Optional[WeightedAggregator] = None,
        config: Optional[ServerConfig] = None,
    ):
        """Initialize SignGuard server.

        Args:
            global_model: Global model
            signature_manager: Signature manager for verification
            detector: Anomaly detector (creates default if None)
            reputation_system: Reputation system (creates default if None)
            aggregator: Aggregator (creates default if None)
            config: Server configuration
        """
        self.global_model = global_model
        self.signature_manager = signature_manager
        
        # Use provided components or create defaults
        self.detector = detector or EnsembleDetector()
        self.reputation_system = reputation_system or DecayReputationSystem()
        self.aggregator = aggregator or WeightedAggregator()
        self.config = config or ServerConfig()
        
        # State
        self.current_round = 0
        self.global_params = self.global_model.state_dict()
        
        # Tracking
        self.excluded_clients: Set[str] = set()
        self.exclusion_reasons: Dict[str, str] = {}
        self.detection_history: List[Dict[str, AnomalyScore]] = []

    def verify_signatures(
        self,
        signed_updates: List[SignedUpdate],
    ) -> tuple[List[SignedUpdate], List[str]]:
        """Verify all signatures.

        Args:
            signed_updates: List of signed updates

        Returns:
            Tuple of (verified_updates, rejected_client_ids)
        """
        verified_updates = []
        rejected_clients = []
        
        for signed_update in signed_updates:
            client_id = signed_update.update.client_id
            
            # Verify signature
            if self.signature_manager.verify_update(signed_update):
                verified_updates.append(signed_update)
            else:
                rejected_clients.append(client_id)
                self.exclusion_reasons[client_id] = "Invalid signature"
        
        return verified_updates, rejected_clients

    def detect_anomalies(
        self,
        verified_updates: List[SignedUpdate],
    ) -> Dict[str, AnomalyScore]:
        """Detect anomalies in verified updates.

        Args:
            verified_updates: Signature-verified updates

        Returns:
            Dictionary mapping client_id -> AnomalyScore
        """
        anomaly_scores = {}
        
        for signed_update in verified_updates:
            client_id = signed_update.update.client_id
            
            # Compute anomaly score
            anomaly_score = self.detector.compute_anomaly_score(
                signed_update.update,
                self.global_params,
            )
            
            anomaly_scores[client_id] = anomaly_score
        
        return anomaly_scores

    def update_reputations(
        self,
        anomaly_scores: Dict[str, AnomalyScore],
        is_verified: Dict[str, bool],
    ) -> None:
        """Update reputations based on anomaly scores.

        Args:
            anomaly_scores: Client anomaly scores
            is_verified: Client signature verification status
        """
        for client_id, anomaly_score in anomaly_scores.items():
            verified = is_verified.get(client_id, True)
            self.reputation_system.update_reputation(
                client_id=client_id,
                anomaly_score=anomaly_score.combined_score,
                round_num=self.current_round,
                is_verified=verified,
            )

    def aggregate(
        self,
        signed_updates: List[SignedUpdate],
    ) -> AggregationResult:
        """Full aggregation pipeline.

        Pipeline:
        1. Verify signatures
        2. Detect anomalies
        3. Update reputations
        4. Aggregate with reputation-weighted averaging

        Args:
            signed_updates: All client updates

        Returns:
            AggregationResult with new global model
        """
        start_time = time.time()
        
        # Reset per-round state
        self.excluded_clients = set()
        self.exclusion_reasons = {}
        
        # Step 1: Verify signatures
        verified_updates, signature_rejected = self.verify_signatures(signed_updates)
        
        for client_id in signature_rejected:
            self.excluded_clients.add(client_id)
        
        # Check minimum participants
        if len(verified_updates) < self.config.min_clients_required:
            raise ValueError(
                f"Insufficient verified updates: {len(verified_updates)} "
                f"< {self.config.min_clients_required}"
            )
        
        # Step 2: Update detector statistics
        self.detector.update_statistics(
            [u.update for u in verified_updates],
            self.global_params,
        )
        
        # Step 3: Detect anomalies
        anomaly_scores = self.detect_anomalies(verified_updates)
        
        # Filter by anomaly threshold
        valid_updates = []
        is_verified = {}
        
        for signed_update in verified_updates:
            client_id = signed_update.update.client_id
            anomaly_score = anomaly_scores[client_id]
            
            # Check if anomalous
            if self.detector.is_anomalous(anomaly_score):
                self.excluded_clients.add(client_id)
                self.exclusion_reasons[client_id] = (
                    f"High anomaly score: {anomaly_score.combined_score:.3f}"
                )
            else:
                valid_updates.append(signed_update)
                is_verified[client_id] = True
        
        # Check minimum participants after anomaly filtering
        if len(valid_updates) < self.config.min_clients_required:
            raise ValueError(
                f"Insufficient valid updates after anomaly detection: "
                f"{len(valid_updates)} < {self.config.min_clients_required}"
            )
        
        # Step 4: Update reputations (for all verified, including anomalous)
        self.update_reputations(anomaly_scores, is_verified)
        
        # Get current reputations
        reputations = self.reputation_system.get_all_reputations()
        
        # Step 5: Aggregate
        result = self.aggregator.aggregate(
            valid_updates,
            reputations,
            self.global_params,
        )
        
        # Update metadata
        result.round_num = self.current_round
        result.execution_time = time.time() - start_time
        result.metadata["exclusion_reasons"] = self.exclusion_reasons
        result.metadata["anomaly_scores"] = {
            client_id: {
                "combined": score.combined_score,
                "magnitude": score.magnitude_score,
                "direction": score.direction_score,
                "loss": score.loss_score,
            }
            for client_id, score in anomaly_scores.items()
        }
        
        # Update global model
        self.update_global_model(result)
        
        # Store detection history
        self.detection_history.append(anomaly_scores)
        
        # Increment round
        self.current_round += 1
        
        return result

    def update_global_model(self, result: AggregationResult) -> None:
        """Update server's global model.

        Args:
            result: Aggregation result
        """
        self.global_model.load_state_dict(result.global_model)
        self.global_params = result.global_model

    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters.

        Returns:
            Global model parameters
        """
        return self.global_params.copy()

    def get_reputations(self) -> Dict[str, float]:
        """Get all client reputations.

        Returns:
            Dictionary mapping client_id -> reputation
        """
        return self.reputation_system.get_all_reputations()

    def get_client_reputation(self, client_id: str) -> float:
        """Get reputation for a specific client.

        Args:
            client_id: Client identifier

        Returns:
            Reputation value
        """
        return self.reputation_system.get_reputation(client_id)

    def initialize_clients(self, client_ids: List[str]) -> None:
        """Initialize reputations for new clients.

        Args:
            client_ids: List of client identifiers
        """
        for client_id in client_ids:
            self.reputation_system.initialize_client(client_id)

    def get_statistics(self) -> Dict:
        """Get server statistics.

        Returns:
            Dictionary with statistics
        """
        reputations = self.get_reputations()
        
        return {
            "current_round": self.current_round,
            "num_clients": len(reputations),
            "avg_reputation": sum(reputations.values()) / len(reputations) if reputations else 0.0,
            "detector_stats": self.detector.get_detector_statistics(),
        }

    def save_checkpoint(
        self,
        filepath: str,
    ) -> None:
        """Save server checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        from src.defenses.signguard_full.legacy.utils.serialization import save_checkpoint
        save_checkpoint(
            round_num=self.current_round,
            global_model=self.global_model,
            reputations=self.get_reputations(),
            filepath=filepath,
        )

    def load_checkpoint(
        self,
        filepath: str,
    ) -> None:
        """Load server checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        from src.defenses.signguard_full.legacy.utils.serialization import load_checkpoint
        checkpoint = load_checkpoint(filepath, model=self.global_model)
        
        self.current_round = checkpoint["round_num"]
        self.global_params = checkpoint["global_model"]
        
        # Restore reputations
        for client_id, reputation in checkpoint["reputations"].items():
            if client_id not in self.reputation_system.reputations:
                self.reputation_system.initialize_client(client_id)
            self.reputation_system.reputations[client_id].reputation = reputation
            self.reputation_system.reputations[client_id].last_update_round = self.current_round
