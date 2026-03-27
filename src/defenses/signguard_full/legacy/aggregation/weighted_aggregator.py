"""Reputation-weighted aggregation."""

from src.defenses.signguard_full.legacy.core.types import SignedUpdate, AggregationResult
from typing import List, Dict
import torch


class WeightedAggregator:
    """Reputation-weighted aggregation of model updates."""

    def __init__(self, min_reputation_threshold: float = 0.1):
        """Initialize weighted aggregator.

        Args:
            min_reputation_threshold: Minimum reputation to participate
        """
        self.min_reputation_threshold = min_reputation_threshold

    def aggregate(
        self,
        signed_updates: List[SignedUpdate],
        reputations: Dict[str, float],
        global_model: Dict[str, torch.Tensor],
    ) -> AggregationResult:
        """Aggregate updates using reputation-weighted averaging.

        Args:
            signed_updates: List of signed updates
            reputations: Client reputations
            global_model: Current global model (for structure reference)

        Returns:
            AggregationResult with aggregated model and metadata
        """
        import time

        start_time = time.time()

        # Filter by reputation threshold
        valid_updates = [
            u
            for u in signed_updates
            if reputations.get(u.update.client_id, 0.0) >= self.min_reputation_threshold
        ]

        excluded = [
            u.update.client_id
            for u in signed_updates
            if u.update.client_id not in {v.update.client_id for v in valid_updates}
        ]

        if not valid_updates:
            raise ValueError(
                "No valid updates above reputation threshold. "
                f"Min threshold: {self.min_reputation_threshold}"
            )

        # Compute weights
        weights = self._compute_weights(valid_updates, reputations)

        # Aggregate parameters
        aggregated_params = {}
        for param_name in global_model.keys():
            weighted_sum = torch.zeros_like(global_model[param_name])
            for update in valid_updates:
                params = update.update.parameters[param_name]
                weighted_sum += weights[update.update.client_id] * params
            aggregated_params[param_name] = weighted_sum

        # Create result
        result = AggregationResult(
            global_model=aggregated_params,
            participating_clients=[u.update.client_id for u in valid_updates],
            excluded_clients=excluded,
            reputation_updates=reputations.copy(),
            round_num=signed_updates[0].update.round_num if signed_updates else 0,
            execution_time=time.time() - start_time,
        )

        return result

    def _compute_weights(
        self,
        signed_updates: List[SignedUpdate],
        reputations: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute normalized reputation weights.

        Args:
            signed_updates: List of signed updates
            reputations: Client reputations

        Returns:
            Dict mapping client_id -> normalized weight
        """
        # Get reputation values
        rep_values = {
            u.update.client_id: reputations.get(u.update.client_id, 0.0)
            for u in signed_updates
        }

        # Normalize
        total = sum(rep_values.values())
        if total == 0:
            # Equal weights if all reputations are 0
            n = len(signed_updates)
            return {u.update.client_id: 1.0 / n for u in signed_updates}

        return {k: v / total for k, v in rep_values.items()}
