"""
SignGuard Server for Flower Framework

Implements server-side verification, anomaly detection, and reputation tracking.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

import flwr as fl
from flwr.common import Parameters

from ..crypto.key_manager import KeyManager
from ..crypto.signature_handler import SignatureHandler
from ..crypto.batch_verifier import BatchVerifier
from ..detection.anomaly_detector import AnomalyDetector
from ..reputation.reputation_manager import ReputationManager
from ..reputation.weighted_aggregator import WeightedAggregator


class SignGuardServer:
    """
    Flower server with SignGuard defense mechanisms.

    Features:
    - Cryptographic signature verification
    - Multi-factor anomaly detection
    - Dynamic reputation tracking
    - Reputation-weighted aggregation
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize SignGuardServer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()

        # Cryptographic components
        self.key_manager = KeyManager()
        self.signature_handler = SignatureHandler()
        self.batch_verifier = BatchVerifier(max_workers=4)

        # Defense components
        self.anomaly_detector = AnomalyDetector(config)
        self.reputation_manager = ReputationManager(config)
        self.weighted_aggregator = WeightedAggregator(
            reputation_manager=self.reputation_manager,
            config=config
        )

        # Server state
        self.current_round = 0
        self.global_model_parameters: Optional[List[np.ndarray]] = None

        # Statistics tracking
        self.detection_history: List[Dict] = []
        self.aggregation_history: List[Dict] = []

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {}

    def register_client(self, client_id: str, public_key_pem: bytes) -> None:
        """
        Register a client's public key.

        Args:
            client_id: Client identifier
            public_key_pem: Public key in PEM format
        """
        self.key_manager.register_client(client_id, public_key_pem)
        self.reputation_manager.register_client(client_id)

    def verify_updates(self,
                       signed_updates: List[Tuple[str, List[np.ndarray], bytes, float]]) -> Tuple[List[str], List[str]]:
        """
        Verify signatures on client updates.

        Args:
            signed_updates: List of (client_id, update, signature_hex, timestamp) tuples

        Returns:
            Tuple of (valid_client_ids, invalid_client_ids)
        """
        valid_clients = []
        invalid_clients = []

        for client_id, update, signature_hex, timestamp in signed_updates:
            # Convert hex signature back to bytes
            signature = bytes.fromhex(signature_hex)

            # Get public key
            public_key = self.key_manager.get_public_key(client_id)
            if public_key is None:
                invalid_clients.append(client_id)
                continue

            # Verify signature
            is_valid = self.signature_handler.verify_parameters(
                public_key=public_key,
                signature=signature,
                parameters=update,
                round_num=self.current_round,
                timestamp=timestamp
            )

            if is_valid:
                valid_clients.append(client_id)
            else:
                invalid_clients.append(client_id)

        return valid_clients, invalid_clients

    def detect_anomalies(self,
                         updates: Dict[str, List[np.ndarray]],
                         global_update: Optional[List[np.ndarray]] = None) -> Dict[str, Dict]:
        """
        Detect anomalies in client updates.

        Args:
            updates: Dictionary mapping client_id to model update
            global_update: Optional global model update

        Returns:
            Dictionary mapping client_id to anomaly results
        """
        results = self.anomaly_detector.detect_anomalies(
            updates=updates,
            global_update=global_update,
            layer_names=None  # Auto-detect from model
        )

        # Update reputations based on anomaly scores
        anomaly_scores = {
            client_id: result['combined']
            for client_id, result in results.items()
        }
        self.reputation_manager.batch_update(anomaly_scores)

        # Store in history
        self.detection_history.append({
            'round': self.current_round,
            'results': results,
            'summary': self.anomaly_detector.get_anomaly_summary(results)
        })

        return results

    def aggregate_updates(self,
                          updates: Dict[str, List[np.ndarray]]) -> Tuple[List[np.ndarray], Dict]:
        """
        Aggregate updates using reputation-weighted averaging.

        Args:
            updates: Dictionary mapping client_id to model update

        Returns:
            Tuple of (aggregated_update, metadata)
        """
        # Check if aggregation is possible
        client_ids = list(updates.keys())
        can_aggregate, reason = self.weighted_aggregator.can_aggregate(client_ids)

        if not can_aggregate:
            print(f"Cannot aggregate: {reason}")
            # Fallback to uniform averaging
            return self._uniform_aggregate(updates)

        # Reputation-weighted aggregation
        aggregated, metadata = self.weighted_aggregator.aggregate_updates(updates)

        # Store in history
        self.aggregation_history.append({
            'round': self.current_round,
            'metadata': metadata
        })

        return aggregated, metadata

    def _uniform_aggregate(self, updates: Dict[str, List[np.ndarray]]) -> Tuple[List[np.ndarray], Dict]:
        """
        Fallback to uniform averaging.

        Args:
            updates: Dictionary mapping client_id to model update

        Returns:
            Tuple of (aggregated_update, metadata)
        """
        client_ids = list(updates.keys())

        # Simple average
        first_update = updates[client_ids[0]]
        aggregated = []

        for layer_idx in range(len(first_update)):
            layer_sum = sum(updates[cid][layer_idx] for cid in client_ids)
            aggregated.append(layer_sum / len(client_ids))

        metadata = {
            'method': 'uniform',
            'num_clients': len(client_ids)
        }

        return aggregated, metadata

    def process_round(self,
                      signed_updates: List[Tuple[str, List[np.ndarray], bytes, float, int]]) -> Tuple[List[np.ndarray], Dict]:
        """
        Process a complete FL round: verify, detect, aggregate.

        Args:
            signed_updates: List of (client_id, update, signature_hex, timestamp, num_examples) tuples

        Returns:
            Tuple of (aggregated_update, round_metadata)
        """
        print(f"\n=== Round {self.current_round} ===")

        # Step 1: Verify signatures
        print("Verifying signatures...")
        updates_dict = {}
        for client_id, update, signature_hex, timestamp, num_examples in signed_updates:
            updates_dict[client_id] = update

        valid_clients, invalid_clients = self.verify_updates([
            (cid, upd, sig, ts)
            for cid, upd, sig, ts, _ in signed_updates
        ])

        print(f"Valid signatures: {len(valid_clients)}/{len(signed_updates)}")
        if invalid_clients:
            print(f"Invalid clients: {invalid_clients}")

        # Filter to valid updates
        valid_updates = {
            cid: updates_dict[cid]
            for cid in valid_clients
        }

        # Step 2: Detect anomalies
        print("Detecting anomalies...")
        anomaly_results = self.detect_anomalies(
            updates=valid_updates,
            global_update=self.global_model_parameters
        )

        summary = self.anomaly_detector.get_anomaly_summary(anomaly_results)
        print(f"Anomaly summary: {summary['num_anomalous']}/{summary['num_clients']} anomalous")
        print(f"Mean anomaly score: {summary['mean_score']:.3f}")

        # Step 3: Aggregate
        print("Aggregating updates...")
        aggregated, metadata = self.aggregate_updates(valid_updates)

        print(f"Aggregated {metadata['num_clients']} updates")
        if 'min_weight' in metadata and 'max_weight' in metadata:
            print(f"Weight range: [{metadata['min_weight']:.3f}, {metadata['max_weight']:.3f}]")

        # Update global model
        self.global_model_parameters = aggregated

        # Round metadata
        round_metadata = {
            'round': self.current_round,
            'num_clients': len(signed_updates),
            'valid_clients': len(valid_clients),
            'invalid_clients': len(invalid_clients),
            'num_anomalous': summary['num_anomalous'],
            'mean_anomaly_score': summary['mean_score'],
            'aggregation_metadata': metadata
        }

        # Increment round
        self.current_round += 1

        return aggregated, round_metadata

    def get_reputations(self) -> Dict[str, float]:
        """Get all client reputations."""
        return self.reputation_manager.get_all_reputations()

    def get_reputation_stats(self) -> Dict:
        """Get reputation statistics."""
        return self.reputation_manager.get_reputation_stats()

    def get_detection_history(self) -> List[Dict]:
        """Get detection history."""
        return self.detection_history

    def get_aggregation_history(self) -> List[Dict]:
        """Get aggregation history."""
        return self.aggregation_history

    def reset(self) -> None:
        """Reset server state."""
        self.current_round = 0
        self.global_model_parameters = None
        self.anomaly_detector.reset()
        self.reputation_manager.reset_all()
        self.detection_history.clear()
        self.aggregation_history.clear()

    def set_global_model(self, parameters: List[np.ndarray]) -> None:
        """Set global model parameters."""
        self.global_model_parameters = parameters

    def get_global_model(self) -> Optional[List[np.ndarray]]:
        """Get global model parameters."""
        return self.global_model_parameters
