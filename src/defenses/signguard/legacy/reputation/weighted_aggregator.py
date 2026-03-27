"""
Weighted Aggregator for SignGuard

Aggregates model updates using reputation weights.
Implements reputation-weighted averaging.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .reputation_manager import ReputationManager


class WeightedAggregator:
    """
    Aggregates federated learning updates using reputation weights.

    Formula:
    θ_global = Σ (w_i * θ_i) / Σ w_i

    Where w_i is the reputation of client i.
    """

    def __init__(self,
                 reputation_manager: Optional[ReputationManager] = None,
                 config: Optional[dict] = None):
        """
        Initialize WeightedAggregator.

        Args:
            reputation_manager: ReputationManager instance
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.reputation_manager = reputation_manager or ReputationManager(config)

        # Aggregation parameters
        rep_config = self.config.get('reputation', {})
        self.min_total_reputation = rep_config.get('min_total_reputation', 0.3)
        self.min_reputation_for_aggregation = rep_config.get('min_reputation_for_aggregation', 0.05)

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'reputation': {
                'min_total_reputation': 0.3,
                'min_reputation_for_aggregation': 0.05
            }
        }

    def compute_weights(self,
                       client_ids: List[str],
                       reputations: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute normalized aggregation weights from reputations.

        Args:
            client_ids: List of client IDs
            reputations: Optional reputation dictionary (uses manager if None)

        Returns:
            Dictionary mapping client_id to normalized weight
        """
        # Get reputations
        if reputations is None:
            reputations = self.reputation_manager.get_all_reputations()

        # Get effective weights (accounts for probation)
        effective_weights = {
            client_id: self.reputation_manager.get_effective_weight(client_id)
            for client_id in client_ids
        }

        # Filter out clients below minimum threshold
        filtered_weights = {
            cid: w
            for cid, w in effective_weights.items()
            if w >= self.min_reputation_for_aggregation
        }

        # Normalize weights to sum to 1
        total_weight = sum(filtered_weights.values())

        if total_weight == 0:
            # Fallback: uniform weights
            num_clients = len(client_ids)
            return {cid: 1.0 / num_clients for cid in client_ids}

        normalized_weights = {
            cid: w / total_weight
            for cid, w in filtered_weights.items()
        }

        return normalized_weights

    def aggregate_updates(self,
                          updates: Dict[str, List[np.ndarray]],
                          reputations: Optional[Dict[str, float]] = None) -> Tuple[List[np.ndarray], Dict]:
        """
        Aggregate model updates using reputation weights.

        Args:
            updates: Dictionary mapping client_id to update (list of arrays)
            reputations: Optional reputation dictionary

        Returns:
            Tuple of (aggregated_update, metadata)
        """
        client_ids = list(updates.keys())

        # Compute weights
        weights = self.compute_weights(client_ids, reputations)

        # Get reference shape from first update
        first_update = updates[client_ids[0]]
        aggregated = []

        # Aggregate layer by layer
        num_layers = len(first_update)

        for layer_idx in range(num_layers):
            # Collect this layer from all clients
            layer_updates = []

            for client_id in client_ids:
                if client_id in weights:
                    layer_updates.append((client_id, updates[client_id][layer_idx]))

            # Weighted average
            layer_shape = layer_updates[0][1].shape
            weighted_sum = np.zeros(layer_shape)

            for client_id, layer_update in layer_updates:
                weight = weights[client_id]
                weighted_sum += weight * layer_update

            aggregated.append(weighted_sum)

        # Metadata
        metadata = {
            'weights': weights,
            'num_clients': len(client_ids),
            'num_filtered': len(client_ids) - len(weights),
            'total_reputation': sum(self.reputation_manager.get_reputation(cid) for cid in client_ids),
            'min_weight': min(weights.values()) if weights else 0.0,
            'max_weight': max(weights.values()) if weights else 0.0,
            'weight_variance': float(np.var(list(weights.values()))) if weights else 0.0
        }

        return aggregated, metadata

    def aggregate_with_num_examples(self,
                                    updates: Dict[str, Tuple[List[np.ndarray], int]],
                                    reputations: Optional[Dict[str, float]] = None) -> Tuple[List[np.ndarray], Dict]:
        """
        Aggregate updates weighted by both reputation and num_examples.

        Common in FL: weight by reputation * num_examples.

        Args:
            updates: Dictionary mapping client_id to (update, num_examples) tuple
            reputations: Optional reputation dictionary

        Returns:
            Tuple of (aggregated_update, metadata)
        """
        client_ids = list(updates.keys())

        # Compute reputation weights
        rep_weights = self.compute_weights(client_ids, reputations)

        # Compute example counts
        total_examples = sum(num_examples for _, num_examples in updates.values())

        # Combined weights: reputation * (num_examples / total_examples)
        combined_weights = {}
        for client_id in client_ids:
            update, num_examples = updates[client_id]
            example_weight = num_examples / total_examples if total_examples > 0 else 1.0 / len(client_ids)
            combined_weights[client_id] = rep_weights[client_id] * example_weight

        # Normalize
        total_combined = sum(combined_weights.values())
        if total_combined > 0:
            combined_weights = {
                cid: w / total_combined
                for cid, w in combined_weights.items()
            }

        # Aggregate
        first_update = updates[client_ids[0]][0]
        aggregated = []

        for layer_idx in range(len(first_update)):
            layer_shape = first_update[layer_idx].shape
            weighted_sum = np.zeros(layer_shape)

            for client_id in client_ids:
                if client_id in combined_weights:
                    update, _ = updates[client_id]
                    weight = combined_weights[client_id]
                    weighted_sum += weight * update[layer_idx]

            aggregated.append(weighted_sum)

        # Metadata
        metadata = {
            'reputation_weights': rep_weights,
            'combined_weights': combined_weights,
            'num_clients': len(client_ids),
            'total_examples': total_examples
        }

        return aggregated, metadata

    def can_aggregate(self, client_ids: List[str]) -> Tuple[bool, str]:
        """
        Check if aggregation is possible with given clients.

        Args:
            client_ids: List of client IDs

        Returns:
            Tuple of (can_aggregate, reason)
        """
        # Compute total reputation
        total_rep = sum(
            self.reputation_manager.get_reputation(cid)
            for cid in client_ids
        )

        if total_rep < self.min_total_reputation:
            return False, f"Total reputation {total_rep:.3f} below minimum {self.min_total_reputation}"

        # Check if any clients meet minimum
        valid_clients = [
            cid for cid in client_ids
            if self.reputation_manager.get_effective_weight(cid) >= self.min_reputation_for_aggregation
        ]

        if len(valid_clients) == 0:
            return False, "No clients meet minimum reputation threshold"

        return True, "OK"

    def get_aggregation_stats(self) -> Dict:
        """
        Get statistics about aggregation weights.

        Returns:
            Dictionary with statistics
        """
        reputations = self.reputation_manager.get_all_reputations()
        weights = self.compute_weights(list(reputations.keys()))

        if len(weights) == 0:
            return {
                'num_clients': 0,
                'total_reputation': 0.0,
                'min_weight': 0.0,
                'max_weight': 0.0,
                'weight_entropy': 0.0
            }

        # Compute weight entropy (measure of concentration)
        weight_values = list(weights.values())
        entropy = -sum(w * np.log(w + 1e-10) for w in weight_values if w > 0)

        return {
            'num_clients': len(weights),
            'total_reputation': sum(reputations.values()),
            'min_weight': min(weight_values),
            'max_weight': max(weight_values),
            'mean_weight': np.mean(weight_values),
            'std_weight': np.std(weight_values),
            'weight_entropy': float(entropy)
        }


def aggregate_updates(updates: List[np.ndarray],
                      weights: List[float]) -> np.ndarray:
    """
    Standalone function to aggregate updates with weights.

    Args:
        updates: List of model updates (numpy arrays)
        weights: List of weights (must sum to 1)

    Returns:
        Aggregated update
    """
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    # Weighted sum
    aggregated = sum(w * u for w, u in zip(weights, updates))

    return aggregated


def compute_weights(reputations: Dict[str, float],
                    min_reputation: float = 0.01) -> Dict[str, float]:
    """
    Standalone function to compute normalized weights from reputations.

    Args:
        reputations: Dictionary of client reputations
        min_reputation: Minimum reputation for inclusion

    Returns:
        Dictionary of normalized weights
    """
    # Filter
    filtered = {
        cid: rep
        for cid, rep in reputations.items()
        if rep >= min_reputation
    }

    # Normalize
    total = sum(filtered.values())
    if total == 0:
        # Uniform weights
        num = len(reputations)
        return {cid: 1.0 / num for cid in reputations}

    normalized = {
        cid: rep / total
        for cid, rep in filtered.items()
    }

    return normalized
