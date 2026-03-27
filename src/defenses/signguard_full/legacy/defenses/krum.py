"""Krum robust aggregation defense."""

import torch
from typing import List
import itertools

from src.defenses.signguard_full.legacy.core.types import ModelUpdate, AggregationResult
from src.defenses.signguard_full.legacy.utils.serialization import parameters_to_vector


class KrumDefense:
    """Krum robust aggregation defense.

    Selects the update closest to its neighbors as the aggregated result.
    Robust against up to f Byzantine clients where f < (n-3)/2.
    """

    def __init__(
        self,
        num_byzantines: int,
        multi_krum: bool = False,
    ):
        """Initialize Krum defense.

        Args:
            num_byzantines: Number of Byzantine clients (f)
            multi_krum: Whether to use multi-krum (average top updates)
        """
        self.num_byzantines = num_byzantines
        self.multi_krum = multi_krum

    def aggregate(
        self,
        updates: List[ModelUpdate],
        global_model: dict[str, torch.Tensor],
    ) -> AggregationResult:
        """Aggregate using Krum algorithm.

        Args:
            updates: List of model updates
            global_model: Current global model (for structure)

        Returns:
            AggregationResult
        """
        import time
        start_time = time.time()
        
        num_clients = len(updates)
        num_closest = num_clients - self.num_byzantines - 2
        
        if num_closest <= 0:
            raise ValueError(
                f"Invalid num_byzantines: {self.num_byzantines}. "
                f"Need n > 2f + 2, where n={num_clients}, f={self.num_byzantines}"
            )
        
        # Convert updates to vectors
        update_vectors = []
        for update in updates:
            vector = parameters_to_vector(update.parameters)
            update_vectors.append(vector)
        
        # Compute pairwise distances
        distances = torch.zeros(num_clients, num_clients)
        for i, j in itertools.combinations(range(num_clients), 2):
            dist = torch.norm(update_vectors[i] - update_vectors[j])
            distances[i, j] = dist
            distances[j, i] = dist
        
        # Compute scores (sum of distances to closest neighbors)
        scores = torch.zeros(num_clients)
        for i in range(num_clients):
            # Get distances to all other clients
            dist_to_others = distances[i]
            # Sort and sum the num_closest smallest
            sorted_dist, _ = torch.sort(dist_to_others)
            scores[i] = sorted_dist[1:num_closest + 1].sum()  # Skip self (distance=0)
        
        if self.multi_krum:
            # Multi-Krum: average top (f+1) updates
            num_selected = self.num_byzantines + 1
            _, top_indices = torch.topk(scores, num_selected, largest=False)
            
            # Average the selected updates
            selected_params = []
            for idx in top_indices:
                selected_params.append(updates[idx.item()].parameters)
            
            aggregated_params = self._average_params(selected_params)
            participating = [updates[i.item()].client_id for i in top_indices]
            
        else:
            # Standard Krum: select the best
            best_idx = torch.argmin(scores)
            
            aggregated_params = updates[best_idx.item()].parameters
            participating = [updates[best_idx.item()].client_id]
        
        # Create result
        result = AggregationResult(
            global_model=aggregated_params,
            participating_clients=participating,
            excluded_clients=[u.client_id for u in updates if u.client_id not in participating],
            reputation_updates={},  # Krum doesn't use reputations
            round_num=updates[0].round_num if updates else 0,
            execution_time=time.time() - start_time,
        )
        
        return result

    def _average_params(
        self,
        param_list: List[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Average parameter dictionaries.

        Args:
            param_list: List of parameter dictionaries

        Returns:
            Averaged parameters
        """
        if not param_list:
            return {}
        
        # Average all parameters
        avg_params = {}
        for param_name in param_list[0].keys():
            stacked = torch.stack([p[param_name] for p in param_list])
            avg_params[param_name] = stacked.mean(dim=0)
        
        return avg_params
