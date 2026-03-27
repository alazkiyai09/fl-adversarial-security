"""Bulyan robust aggregation defense."""

import torch
from typing import List
import itertools

from src.defenses.signguard_full.legacy.core.types import ModelUpdate, AggregationResult
from src.defenses.signguard_full.legacy.utils.serialization import parameters_to_vector


class BulyanDefense:
    """Bulyan robust aggregation defense.

    Combines Krum selection with coordinate-wise trimmed mean.
    More robust than either method alone.
    """

    def __init__(
        self,
        num_byzantines: int,
        distance_metric: str = "euclidean",
    ):
        """Initialize Bulyan defense.

        Args:
            num_byzantines: Number of Byzantine clients (f)
            distance_metric: Distance metric for Krum
        """
        self.num_byzantines = num_byzantines
        self.distance_metric = distance_metric

    def aggregate(
        self,
        updates: List[ModelUpdate],
        global_model: dict[str, torch.Tensor],
    ) -> AggregationResult:
        """Aggregate using Bulyan algorithm.

        Args:
            updates: List of model updates
            global_model: Current global model (for structure)

        Returns:
            AggregationResult
        """
        import time
        start_time = time.time()
        
        num_clients = len(updates)
        
        # Bulyan requires n > 4f
        if num_clients <= 4 * self.num_byzantines:
            raise ValueError(
                f"Bulyan requires n > 4f, where n={num_clients}, f={self.num_byzantines}"
            )
        
        # Step 1: Use Krum to select 2f+1 candidate updates
        num_krum_points = 2 * self.num_byzantines + 1
        
        # Compute scores for all updates (using Krum logic)
        scores = self._compute_krum_scores(updates)
        
        # Select top 2f+1 updates
        _, top_indices = torch.topk(scores, num_krum_points, largest=False)
        selected_updates = [updates[i.item()] for i in top_indices]
        
        # Step 2: Apply coordinate-wise trimmed mean on selected updates
        aggregated_params = self._trimmed_mean(selected_updates)
        
        # Create result
        participating = [updates[i.item()].client_id for i in top_indices]
        excluded = [u.client_id for u in updates if u.client_id not in participating]
        
        result = AggregationResult(
            global_model=aggregated_params,
            participating_clients=participating,
            excluded_clients=excluded,
            reputation_updates={},
            round_num=updates[0].round_num if updates else 0,
            execution_time=time.time() - start_time,
        )
        
        return result

    def _compute_krum_scores(
        self,
        updates: List[ModelUpdate],
    ) -> torch.Tensor:
        """Compute Krum-style scores for updates.

        Args:
            updates: List of model updates

        Returns:
            Tensor of scores
        """
        num_clients = len(updates)
        num_closest = num_clients - self.num_byzantines - 2
        
        # Convert to vectors
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
        
        # Compute scores
        scores = torch.zeros(num_clients)
        for i in range(num_clients):
            dist_to_others = distances[i]
            sorted_dist, _ = torch.sort(dist_to_others)
            scores[i] = sorted_dist[1:num_closest + 1].sum()
        
        return scores

    def _trimmed_mean(
        self,
        updates: List[ModelUpdate],
    ) -> dict[str, torch.Tensor]:
        """Apply coordinate-wise trimmed mean.

        Args:
            updates: List of selected updates

        Returns:
            Trimmed mean parameters
        """
        num_updates = len(updates)
        num_trim = self.num_byzantines
        
        aggregated_params = {}
        
        for param_name in updates[0].parameters.keys():
            # Stack parameters
            stacked = torch.stack([u.parameters[param_name] for u in updates])
            
            # Sort along client dimension and trim
            sorted_stack, _ = torch.sort(stacked, dim=0)
            trimmed_stack = sorted_stack[num_trim:num_updates - num_trim]
            
            # Average
            aggregated_params[param_name] = trimmed_stack.mean(dim=0)
        
        return aggregated_params
