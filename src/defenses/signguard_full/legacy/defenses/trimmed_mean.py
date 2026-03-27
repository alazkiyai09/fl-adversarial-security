"""Trimmed Mean robust aggregation defense."""

import torch
from typing import List

from src.defenses.signguard_full.legacy.core.types import ModelUpdate, AggregationResult


class TrimmedMeanDefense:
    """Trimmed Mean robust aggregation defense.

    Trims the smallest and largest parameter values and averages the rest.
    Robust against up to f Byzantine clients.
    """

    def __init__(
        self,
        trim_ratio: float = 0.2,
        per_parameter: bool = False,
    ):
        """Initialize Trimmed Mean defense.

        Args:
            trim_ratio: Fraction to trim from each end [0, 0.5)
            per_parameter: Whether to trim per-parameter or globally
        """
        if not 0 <= trim_ratio < 0.5:
            raise ValueError(f"trim_ratio must be in [0, 0.5), got {trim_ratio}")
        
        self.trim_ratio = trim_ratio
        self.per_parameter = per_parameter

    def aggregate(
        self,
        updates: List[ModelUpdate],
        global_model: dict[str, torch.Tensor],
    ) -> AggregationResult:
        """Aggregate using trimmed mean.

        Args:
            updates: List of model updates
            global_model: Current global model (for structure)

        Returns:
            AggregationResult
        """
        import time
        start_time = time.time()
        
        if self.per_parameter:
            aggregated_params = self._aggregate_per_parameter(updates)
        else:
            aggregated_params = self._aggregate_global(updates)
        
        # Create result
        result = AggregationResult(
            global_model=aggregated_params,
            participating_clients=[u.client_id for u in updates],
            excluded_clients=[],
            reputation_updates={},
            round_num=updates[0].round_num if updates else 0,
            execution_time=time.time() - start_time,
        )
        
        return result

    def _aggregate_global(
        self,
        updates: List[ModelUpdate],
    ) -> dict[str, torch.Tensor]:
        """Aggregate with global trimming.

        Args:
            updates: List of model updates

        Returns:
            Aggregated parameters
        """
        # Compute L2 norms of all updates
        from src.defenses.signguard_full.legacy.utils.serialization import parameters_to_vector
        
        norms = []
        for update in updates:
            vector = parameters_to_vector(update.parameters)
            norm = torch.norm(vector)
            norms.append(norm)
        
        norms = torch.tensor(norms)
        num_updates = len(updates)
        num_trim = int(num_updates * self.trim_ratio)
        
        # Get indices to keep (not trimmed)
        _, sorted_indices = torch.sort(norms)
        keep_indices = sorted_indices[num_trim:num_updates - num_trim]
        
        # Average the kept updates
        kept_params = [updates[i.item()].parameters for i in keep_indices]
        return self._average_params(kept_params)

    def _aggregate_per_parameter(
        self,
        updates: List[ModelUpdate],
    ) -> dict[str, torch.Tensor]:
        """Aggregate with per-parameter trimming.

        Args:
            updates: List of model updates

        Returns:
            Aggregated parameters
        """
        aggregated_params = {}
        num_updates = len(updates)
        num_trim = int(num_updates * self.trim_ratio)
        
        for param_name in updates[0].parameters.keys():
            # Stack all parameters
            stacked = torch.stack([u.parameters[param_name] for u in updates])
            
            # Flatten for sorting
            original_shape = stacked.shape
            flat = stacked.view(num_updates, -1)
            
            # Sort and trim along the client dimension
            sorted_flat, _ = torch.sort(flat, dim=0)
            trimmed_flat = sorted_flat[num_trim:num_updates - num_trim]
            
            # Average
            averaged_flat = trimmed_flat.mean(dim=0)
            
            # Restore shape
            aggregated_params[param_name] = averaged_flat.view(original_shape[1:])
        
        return aggregated_params

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
        
        avg_params = {}
        for param_name in param_list[0].keys():
            stacked = torch.stack([p[param_name] for p in param_list])
            avg_params[param_name] = stacked.mean(dim=0)
        
        return avg_params
