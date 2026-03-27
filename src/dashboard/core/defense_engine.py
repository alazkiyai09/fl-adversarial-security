"""
Defense mechanisms for FL security.
Implements SignGuard, Krum, FoolsGold, and other defenses.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from .data_models import (
    DefenseConfig,
    ClientMetric,
    SecurityEvent
)


class DefenseEngine:
    """
    Implements various defense mechanisms against FL attacks.
    """

    def __init__(self, config: DefenseConfig, num_clients: int):
        """
        Initialize defense engine.

        Args:
            config: Defense configuration
            num_clients: Total number of clients in the system
        """
        self.config = config
        self.num_clients = num_clients

        # Reputation tracking (for SignGuard, FoolsGold)
        self.reputation_scores: Dict[int, float] = {
            i: 1.0 for i in range(num_clients)
        }

        # Historical updates for trend analysis
        self.update_history: Dict[int, List[np.ndarray]] = defaultdict(list)

        # Defense activation log
        self.defense_log: List[Dict] = []

    def apply_defense(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray],
        client_metrics: Dict[int, ClientMetric]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        Apply configured defense mechanism.

        Args:
            round_num: Current round number
            client_updates: Dictionary mapping client_id to update vector
            client_metrics: Dictionary mapping client_id to client metrics

        Returns:
            Tuple of (filtered_updates, security_events)
        """
        if self.config.defense_type == "none":
            return client_updates, []

        # Update history with current updates
        for client_id, update in client_updates.items():
            self.update_history[client_id].append(update)
            # Keep only recent updates (window size)
            window_size = getattr(self.config, 'signguard_window_size', 5)
            if len(self.update_history[client_id]) > window_size:
                self.update_history[client_id].pop(0)

        # Apply specific defense
        if self.config.defense_type == "signguard":
            return self._apply_signguard(round_num, client_updates)
        elif self.config.defense_type == "krum":
            return self._apply_krum(round_num, client_updates)
        elif self.config.defense_type == "foolsgold":
            return self._apply_foolsgold(round_num, client_updates)
        elif self.config.defense_type == "trim_mean":
            return self._apply_trim_mean(round_num, client_updates)
        elif self.config.defense_type == "median":
            return self._apply_median(round_num, client_updates)
        else:
            return client_updates, []

    def _apply_signguard(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        SignGuard: Detect anomalies by analyzing update sign patterns.
        Downweights clients with inconsistent sign patterns.
        """
        events = []
        modified_updates = {}

        if len(client_updates) < 2:
            return client_updates, events

        # Compute sign consistency scores
        client_ids = list(client_updates.keys())
        first_client = client_ids[0]
        reference_update = client_updates[first_client]
        reference_signs = np.sign(reference_update)

        anomaly_scores = {}
        for client_id in client_ids:
            update = client_updates[client_id]
            signs = np.sign(update)

            # Sign agreement rate
            agreement = np.mean(signs == reference_signs)
            anomaly_score = 1.0 - agreement
            anomaly_scores[client_id] = anomaly_score

            # Update reputation using exponential moving average
            decay = self.config.signguard_decay_factor
            self.reputation_scores[client_id] = (
                decay * self.reputation_scores[client_id] +
                (1 - decay) * (1.0 - anomaly_score)
            )

        # Detect and flag anomalies
        threshold = self.config.anomaly_threshold
        for client_id, score in anomaly_scores.items():
            if score > threshold:
                events.append(SecurityEvent(
                    event_id=f"sg_{round_num}_{client_id}",
                    event_type="defense_activated",
                    severity="medium",
                    message=f"SignGuard flagged Client {client_id} (anomaly score: {score:.3f})",
                    round_num=round_num,
                    affected_clients=[client_id],
                    confidence=score
                ))

        # Apply downweighting
        for client_id, update in client_updates.items():
            weight = self.reputation_scores.get(client_id, 1.0)

            # If action is drop and reputation is too low, exclude
            if (self.config.action_on_detection == "drop" and
                weight < self.config.reputation_threshold):
                continue

            # Otherwise, downweight the update
            if self.config.action_on_detection == "downweight":
                modified_updates[client_id] = update * weight
            else:
                modified_updates[client_id] = update

        # Log defense action
        self.defense_log.append({
            "round": round_num,
            "defense": "signguard",
            "anomalies_detected": len(events),
            "clients_excluded": len(client_updates) - len(modified_updates)
        })

        return modified_updates, events

    def _apply_krum(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        Krum: Select update closest to others (most benign).
        """
        events = []
        client_ids = list(client_updates.keys())
        updates = [client_updates[cid] for cid in client_ids]

        if len(updates) <= 2:
            return client_updates, events

        # Compute pairwise distances
        n = len(updates)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(updates[i] - updates[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Krum score: sum of closest n-f-2 distances
        f = self.config.krum_num_attackers
        krum_scores = []
        for i in range(n):
            # Get distances to other updates
            row = distances[i]
            # Sort and sum closest (excluding self)
            closest = np.sort(row)[:n - f - 1]
            krum_scores.append(np.sum(closest))

        # Select update with minimum Krum score
        best_idx = int(np.argmin(krum_scores))
        selected_client = client_ids[best_idx]

        # Flag all other clients as potentially malicious
        for i, client_id in enumerate(client_ids):
            if i != best_idx:
                events.append(SecurityEvent(
                    event_id=f"krum_{round_num}_{client_id}",
                    event_type="defense_activated",
                    severity="low",
                    message=f"Krum excluded Client {client_id} update",
                    round_num=round_num,
                    affected_clients=[client_id],
                    confidence=0.7
                ))

        # Return only the selected update
        modified_updates = {selected_client: client_updates[selected_client]}

        self.defense_log.append({
            "round": round_num,
            "defense": "krum",
            "selected_client": selected_client,
            "excluded_count": len(client_ids) - 1
        })

        return modified_updates, events

    def _apply_foolsgold(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        FoolsGold: Detect Byzantine attacks based on similarity.
        Downweights clients that are too similar to each other.
        """
        events = []
        client_ids = list(client_updates.keys())

        if len(client_ids) < 2:
            return client_updates, events

        # Compute cosine similarities
        n = len(client_ids)
        similarities = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                update_i = client_updates[client_ids[i]]
                update_j = client_updates[client_ids[j]]

                # Cosine similarity
                dot = np.dot(update_i, update_j)
                norm_i = np.linalg.norm(update_i)
                norm_j = np.linalg.norm(update_j)

                if norm_i > 0 and norm_j > 0:
                    sim = dot / (norm_i * norm_j)
                else:
                    sim = 0.0

                similarities[i, j] = sim
                similarities[j, i] = sim

        # Compute FoolsGold weights
        weights = np.ones(n)
        for i in range(n):
            # Sum of similarities with other clients
            similarity_sum = np.sum(similarities[i]) - similarities[i, i]
            # Inverse weighting: clients with high similarity get lower weight
            if similarity_sum > 0:
                weights[i] = 1.0 / (1.0 + similarity_sum)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Apply weights to updates
        modified_updates = {}
        for i, client_id in enumerate(client_ids):
            modified_updates[client_id] = client_updates[client_id] * weights[i]

        # Flag clients with very low weights
        threshold = np.mean(weights) - 0.5 * np.std(weights)
        for i, client_id in enumerate(client_ids):
            if weights[i] < threshold:
                events.append(SecurityEvent(
                    event_id=f"fg_{round_num}_{client_id}",
                    event_type="defense_activated",
                    severity="low",
                    message=f"FoolsGold downweighted Client {client_id} (weight: {weights[i]:.3f})",
                    round_num=round_num,
                    affected_clients=[client_id],
                    confidence=1.0 - weights[i]
                ))

        self.defense_log.append({
            "round": round_num,
            "defense": "foolsgold",
            "weights": {client_ids[i]: float(weights[i]) for i in range(n)},
            "anomalies_detected": len(events)
        })

        return modified_updates, events

    def _apply_trim_mean(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        Trimmed mean: Remove extreme updates and average the rest.
        """
        events = []
        client_ids = list(client_updates.keys())
        n = len(client_ids)

        if n < 3:
            return client_updates, events

        # Stack updates
        updates = np.stack([client_updates[cid] for cid in client_ids])

        # Compute norms to find extremes
        norms = np.linalg.norm(updates, axis=1)

        # Determine number to trim
        k = max(1, int(self.config.trim_ratio * n))

        # Find indices with smallest and largest norms
        extreme_indices = np.argsort(norms)[:k]  # Smallest
        extreme_indices = np.concatenate([
            extreme_indices,
            np.argsort(norms)[-k:]  # Largest
        ])
        extreme_indices = np.unique(extreme_indices)

        # Flag extreme clients
        for idx in extreme_indices:
            client_id = client_ids[idx]
            events.append(SecurityEvent(
                event_id=f"trim_{round_num}_{client_id}",
                event_type="defense_activated",
                severity="low",
                message=f"Trim-mean excluded Client {client_id} (extreme norm: {norms[idx]:.3f})",
                round_num=round_num,
                affected_clients=[client_id],
                confidence=0.6
            ))

        # Return non-excluded updates
        modified_updates = {
            client_ids[i]: client_updates[client_ids[i]]
            for i in range(n)
            if i not in extreme_indices
        }

        self.defense_log.append({
            "round": round_num,
            "defense": "trim_mean",
            "excluded_count": len(extreme_indices)
        })

        return modified_updates, events

    def _apply_median(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        Median aggregation: robust to outliers but doesn't exclude clients.
        """
        events = []
        client_ids = list(client_updates.keys())
        n = len(client_ids)

        if n < 3:
            return client_updates, events

        # Stack updates
        updates = np.stack([client_updates[cid] for cid in client_ids])

        # Compute median along client axis
        median_update = np.median(updates, axis=0)

        # Identify outliers (far from median)
        distances = np.array([
            np.linalg.norm(updates[i] - median_update)
            for i in range(n)
        ])

        threshold = np.median(distances) + 2 * np.std(distances)
        outliers = np.where(distances > threshold)[0]

        # Flag outliers
        for idx in outliers:
            client_id = client_ids[idx]
            events.append(SecurityEvent(
                event_id=f"median_{round_num}_{client_id}",
                event_type="anomaly_detected",
                severity="low",
                message=f"Median aggregation flagged Client {client_id} as outlier",
                round_num=round_num,
                affected_clients=[client_id],
                confidence=0.5
            ))

        # Return all updates (median doesn't exclude, just flags)
        self.defense_log.append({
            "round": round_num,
            "defense": "median",
            "outliers_detected": len(outliers)
        })

        return client_updates, events

    def get_reputation_scores(self) -> Dict[int, float]:
        """Get current reputation scores for all clients."""
        return self.reputation_scores.copy()

    def get_defense_statistics(self) -> Dict:
        """Get statistics about defense activations."""
        return {
            "defense_type": self.config.defense_type,
            "total_activations": len(self.defense_log),
            "recent_reputations": self.reputation_scores.copy(),
            "defense_history": self.defense_log[-10:]  # Last 10 activations
        }
