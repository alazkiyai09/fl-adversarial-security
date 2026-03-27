"""FoolsGold robust aggregation defense."""

import torch
from typing import List, Dict
from collections import defaultdict

from src.defenses.signguard_full.legacy.core.types import ModelUpdate, AggregationResult
from src.defenses.signguard_full.legacy.utils.serialization import parameters_to_vector


class FoolsGoldDefense:
    """FoolsGold robust aggregation defense.

    Uses similarity scores between updates to compute weights.
    Reduces weights for colluding/copying clients.
    """

    def __init__(
        self,
        history_length: int = 10,
        clip_value: float = 0.5,
        min_weight: float = 0.01,
    ):
        """Initialize FoolsGold defense.

        Args:
            history_length: Number of past rounds to consider
            clip_value: Value for clipping similarity scores
            min_weight: Minimum weight for any client
        """
        self.history_length = history_length
        self.clip_value = clip_value
        self.min_weight = min_weight
        self.update_history: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def aggregate(
        self,
        updates: List[ModelUpdate],
        global_model: dict[str, torch.Tensor],
    ) -> AggregationResult:
        """Aggregate using FoolsGold similarity-based weighting.

        Args:
            updates: List of model updates
            global_model: Current global model (for structure)

        Returns:
            AggregationResult
        """
        import time
        start_time = time.time()
        
        # Convert to vectors
        update_vectors = {}
        for update in updates:
            vector = parameters_to_vector(update.parameters)
            update_vectors[update.client_id] = vector
        
        # Update history
        for update in updates:
            client_id = update.client_id
            self.update_history[client_id].append(update_vectors[client_id])
            
            # Trim history
            if len(self.update_history[client_id]) > self.history_length:
                self.update_history[client_id] = self.update_history[client_id][-self.history_length:]
        
        # Compute similarity scores
        weights = self._compute_weights(update_vectors)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Weighted aggregation
        aggregated_params = self._weighted_aggregate(updates, weights)
        
        # Create result
        result = AggregationResult(
            global_model=aggregated_params,
            participating_clients=list(weights.keys()),
            excluded_clients=[],
            reputation_updates=weights,  # Use weights as reputations
            round_num=updates[0].round_num if updates else 0,
            execution_time=time.time() - start_time,
        )
        
        return result

    def _compute_weights(
        self,
        update_vectors: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute FoolsGold weights based on similarity.

        Args:
            update_vectors: Client ID -> update vector

        Returns:
            Client ID -> weight
        """
        client_ids = list(update_vectors.keys())
        num_clients = len(client_ids)
        weights = {}
        
        for client_id in client_ids:
            # Get history for this client
            history = self.update_history.get(client_id, [])
            
            if not history:
                weights[client_id] = 1.0
                continue
            
            # Use latest update
            current_vector = update_vectors[client_id]
            
            # Compute similarity with other clients
            similarities = []
            for other_id in client_ids:
                if other_id == client_id:
                    continue
                
                other_vector = update_vectors[other_id]
                
                # Cosine similarity
                sim = torch.dot(current_vector, other_vector) / (
                    torch.norm(current_vector) * torch.norm(other_vector) + 1e-10
                )
                similarities.append(sim.item())
            
            # Clip similarities
            similarities = [max(-self.clip_value, min(self.clip_value, s)) for s in similarities]
            
            # Average similarity
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Convert to weight (lower similarity -> lower weight)
            # FoolsGold uses the number of colluding clients to reduce weight
            weight = max(self.min_weight, 1.0 / (1.0 + avg_similarity * num_clients))
            weights[client_id] = weight
        
        return weights

    def _weighted_aggregate(
        self,
        updates: List[ModelUpdate],
        weights: Dict[str, float],
    ) -> dict[str, torch.Tensor]:
        """Perform weighted aggregation.

        Args:
            updates: List of model updates
            weights: Client weights

        Returns:
            Aggregated parameters
        """
        aggregated_params = {}
        
        for param_name in updates[0].parameters.keys():
            weighted_sum = torch.zeros_like(updates[0].parameters[param_name])
            
            for update in updates:
                weight = weights.get(update.client_id, 0.0)
                weighted_sum += weight * update.parameters[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
