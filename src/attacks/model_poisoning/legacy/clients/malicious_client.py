"""
Malicious client that performs model poisoning attacks.

Wraps attack strategies and applies them to model updates before sending to server.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
import flwr as fl

from .honest_client import HonestClient
from ..attacks.base_poison import ModelPoisoningAttack


class MaliciousClient(HonestClient):
    """
    Malicious client that performs model poisoning attacks.

    Inherits from HonestClient but applies poisoning strategies to
    model updates before sending them to the server. Supports various
    timing strategies for attacks.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        client_id: int,
        attack_strategy: ModelPoisoningAttack,
        attack_timing: str = "continuous",
        attack_frequency: int = 1,
        late_stage_start: int = 20,
        device: str = "cpu"
    ):
        """
        Initialize malicious client.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            test_loader: Test data loader
            client_id: Unique client identifier
            attack_strategy: Poisoning attack instance
            attack_timing: "continuous", "intermittent", or "late_stage"
            attack_frequency: Attack every N rounds (for intermittent)
            late_stage_start: Start attack after round N (for late_stage)
            device: Device for training
        """
        super().__init__(model, train_loader, test_loader, client_id, device)

        self.attack_strategy = attack_strategy
        self.attack_timing = attack_timing
        self.attack_frequency = attack_frequency
        self.late_stage_start = late_stage_start
        self.current_round = 0

        # Attack tracking
        self.attack_history = []  # List of (round, attacked, attack_info)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model and apply poisoning attack to updates.

        Args:
            parameters: Initial model parameters from server
            config: Training configuration

        Returns:
            Tuple of (poisoned_parameters, num_examples, metrics)
        """
        # Get current round from config
        self.current_round = config.get("server_round", 0)

        # Perform honest training first
        updated_params, num_examples, metrics = super().fit(parameters, config)

        # Apply poisoning attack if conditions met
        should_attack = self.attack_strategy.should_attack(
            current_round=self.current_round,
            timing_strategy=self.attack_timing,
            frequency=self.attack_frequency,
            start_round=self.late_stage_start
        )

        # Flatten parameters for poisoning
        flat_params = np.concatenate([p.flatten() for p in updated_params])
        layer_info = self.model.get_layer_info()

        if should_attack:
            # Apply poisoning
            poisoned_flat = self.attack_strategy.poison_update(flat_params, layer_info)

            # Reshape back to original structure
            poisoned_params = self._reshape_parameters(poisoned_flat, updated_params)

            # Update metrics
            metrics["is_malicious"] = True
            metrics["attack_type"] = self.attack_strategy.attack_name

            attack_info = {
                "round": self.current_round,
                "attacked": True,
                "attack_name": self.attack_strategy.attack_name
            }
        else:
            # No attack this round (honest behavior)
            poisoned_params = updated_params
            metrics["is_malicious"] = False

            attack_info = {
                "round": self.current_round,
                "attacked": False,
                "attack_name": self.attack_strategy.attack_name
            }

        self.attack_history.append(attack_info)
        return poisoned_params, num_examples, metrics

    def _reshape_parameters(
        self,
        flat_params: np.ndarray,
        reference_params: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Reshape flattened parameters back to original structure.

        Args:
            flat_params: Flattened parameter array
            reference_params: Reference for shapes

        Returns:
            List of reshaped parameter arrays
        """
        reshaped = []
        idx = 0

        for ref_param in reference_params:
            size = ref_param.size
            param_slice = flat_params[idx:idx + size]
            reshaped.append(param_slice.reshape(ref_param.shape))
            idx += size

        return reshaped

    def get_attack_history(self) -> List[Dict]:
        """Get history of attack executions."""
        return self.attack_history

    def reset_attack_history(self):
        """Clear attack history."""
        self.attack_history = []
        self.current_round = 0
        self.attack_strategy.reset()
