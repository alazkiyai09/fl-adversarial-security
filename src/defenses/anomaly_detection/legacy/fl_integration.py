"""
Flower integration for real-time anomaly detection during FL training.
Server-side callback to detect malicious clients each round.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import time
from flwr.common import Parameters, FitRes, Scalar
from flwr.server.server import ClientManager
from flwr.server.strategy import FedAvg

from src.defenses.anomaly_detection.legacy.ensemble.voting_ensemble import VotingEnsemble
from src.defenses.anomaly_detection.legacy.utils.updates_parser import extract_updates, flatten_update


class AnomalyDetectionStrategy(FedAvg):
    """
    Federated averaging strategy with anomaly detection.

    Extends Flower's FedAvg to detect and filter malicious clients
    before aggregation.
    """

    def __init__(
        self,
        detection_ensemble: VotingEnsemble,
        *args,
        filter_malicious: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize anomaly detection strategy.

        Args:
            detection_ensemble: Fitted voting ensemble for detection
            filter_malicious: If True, exclude malicious clients from aggregation
            verbose: If True, print detection results each round
        """
        super().__init__(*args, **kwargs)

        self.detection_ensemble = detection_ensemble
        self.filter_malicious = filter_malicious
        self.verbose = verbose

        # Detection statistics
        self.detection_history: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates with anomaly detection.

        Args:
            server_round: Current FL round
            results: List of (client_id, FitRes) tuples
            failures: List of failures

        Returns:
            Aggregated parameters and metrics
        """
        # Extract client updates
        client_updates = {}
        for client_id, fit_res in results:
            client_updates[client_id] = fit_res.parameters

        # Detect malicious clients
        detection_start = time.time()
        malicious_ids, detection_summary = self._detect_malicious_clients(
            client_updates,
            server_round
        )
        detection_time = time.time() - detection_start

        # Log detection results
        if self.verbose:
            print(f"\n=== Round {server_round} Detection Results ===")
            print(f"Malicious clients detected: {malicious_ids}")
            print(f"Detection time: {detection_time*1000:.2f}ms")
            print(f"Detection summary: {detection_summary}")

        # Filter malicious clients if enabled
        if self.filter_malicious:
            filtered_results = [
                (cid, res) for cid, res in results
                if cid not in malicious_ids
            ]

            if self.verbose and malicious_ids:
                print(f"Filtered {len(malicious_ids)} malicious clients")

            results = filtered_results

        # Call parent aggregation with (possibly filtered) results
        aggregated_params, metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        # Add detection metrics to return metrics
        metrics['num_malicious_detected'] = len(malicious_ids)
        metrics['detection_time_ms'] = detection_time * 1000
        metrics['num_clients_total'] = len(client_updates)
        metrics['num_clients_honest'] = len(results)

        # Store detection history
        self.detection_history.append({
            'round': server_round,
            'malicious_ids': malicious_ids,
            'detection_time': detection_time,
            'summary': detection_summary
        })

        return aggregated_params, metrics

    def _detect_malicious_clients(
        self,
        client_updates: Dict[int, Parameters],
        round_num: int
    ) -> Tuple[List[int], Dict]:
        """
        Detect malicious clients using ensemble.

        Args:
            client_updates: Dictionary of client_id -> Parameters
            round_num: Current round number

        Returns:
            Tuple of (malicious_ids, detection_summary)
        """
        malicious_ids = []
        all_scores = {}
        all_votes = {}

        for client_id, params in client_updates.items():
            # Extract and flatten update
            try:
                layers = extract_updates(params)
                flattened = flatten_update(layers)

                # Get ensemble decision
                is_malicious = self.detection_ensemble.is_malicious(flattened)

                if is_malicious:
                    malicious_ids.append(client_id)

                # Get detailed scores for analysis
                scores = self.detection_ensemble.get_individual_scores(flattened)
                all_scores[client_id] = scores

                # Get voting summary
                summary = self.detection_ensemble.get_voting_summary(flattened)
                all_votes[client_id] = summary

            except Exception as e:
                print(f"Warning: Failed to process client {client_id}: {e}")
                continue

        detection_summary = {
            'total_clients': len(client_updates),
            'malicious_count': len(malicious_ids),
            'all_scores': all_scores,
            'all_votes': all_votes
        }

        return malicious_ids, detection_summary


def create_detection_ensemble(config_path: str = "config/detection_config.yaml") -> VotingEnsemble:
    """
    Create detection ensemble from config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Fitted VotingEnsemble instance
    """
    import yaml
    from src.defenses.anomaly_detection.legacy.detectors import (
        MagnitudeDetector,
        SimilarityDetector,
        LayerwiseDetector,
        HistoricalDetector,
        ClusteringDetector,
        SpectralDetector
    )

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create detectors based on config
    detectors = []

    if config['magnitude']['enabled']:
        detectors.append(MagnitudeDetector(
            method=config['magnitude']['method'],
            threshold=config['magnitude']['zscore_threshold']
        ))

    if config['similarity']['enabled']:
        detectors.append(SimilarityDetector(
            similarity_threshold=config['similarity']['similarity_threshold'],
            comparison_target=config['similarity']['comparison_target']
        ))

    if config['layerwise']['enabled']:
        detectors.append(LayerwiseDetector(
            layer_threshold=config['layerwise']['layer_threshold'],
            min_anomalous_layers=config['layerwise']['min_anomalous_layers']
        ))

    if config['historical']['enabled']:
        detectors.append(HistoricalDetector(
            alpha=config['historical']['alpha'],
            threshold=config['historical']['threshold'],
            warmup_rounds=config['historical']['warmup_rounds']
        ))

    if config['clustering']['enabled']:
        detectors.append(ClusteringDetector(
            method=config['clustering']['method'],
            contamination=config['clustering']['contamination']
        ))

    if config['spectral']['enabled']:
        detectors.append(SpectralDetector(
            n_components=config['spectral']['n_components'],
            threshold=config['spectral']['threshold']
        ))

    # Create ensemble
    ensemble = VotingEnsemble(
        detectors=detectors,
        voting=config['ensemble']['voting'],
        weights=config['ensemble'].get('weights')
    )

    return ensemble
