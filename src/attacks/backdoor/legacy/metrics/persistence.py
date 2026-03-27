"""
Backdoor persistence testing.
Tests how long backdoor survives after attacker stops participating.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np

from src.attacks.backdoor.legacy.models.fraud_model import FraudMLP
from src.attacks.backdoor.legacy.metrics.attack_metrics import compute_clean_accuracy, compute_attack_success_rate


def test_backdoor_persistence(
    model: nn.Module,
    clean_test_loader: DataLoader,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    trigger_config: Dict,
    persistence_rounds: List[int],
    server,
    clients: List,
    device: str = 'cpu'
) -> Dict[int, Dict[str, float]]:
    """
    Test backdoor persistence after attacker stops participating.

    Args:
        model: Global model with backdoor
        clean_test_loader: Clean test data
        test_features: Test features for ASR
        test_labels: Test labels
        trigger_config: Trigger configuration
        persistence_rounds: List of rounds to test (e.g., [5, 10, 20])
        server: FL server
        clients: List of honest clients only (attacker removed)
        device: Device to run on

    Returns:
        Dictionary mapping round number to metrics
    """
    persistence_results = {}

    # Save initial state (with backdoor)
    initial_weights = {name: param.data.clone()
                      for name, param in model.named_parameters()}

    for n_rounds in persistence_rounds:
        print(f"\nTesting persistence after {n_rounds} rounds without attacker...")

        # Restore initial state
        for name, param in model.named_parameters():
            param.data = initial_weights[name].clone()

        server.global_weights = model.get_weights()

        # Train for N rounds without attacker
        for round_idx in range(n_rounds):
            # Define client training function (only honest clients)
            def client_train_fn(client_id, global_weights):
                return clients[client_id].train(global_weights)

            # Train one round
            server.fit_round(round_idx, client_train_fn)

        # Evaluate
        clean_acc = compute_clean_accuracy(model, clean_test_loader, device)
        asr = compute_attack_success_rate(
            model, test_features, test_labels, trigger_config,
            device=device
        )

        persistence_results[n_rounds] = {
            'clean_accuracy': clean_acc,
            'attack_success_rate': asr
        }

        print(f"  Clean accuracy: {clean_acc:.4f}")
        print(f"  ASR: {asr:.4f}")

    return persistence_results


def track_asr_over_rounds(
    model: nn.Module,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    trigger_config: Dict,
    n_rounds: int,
    server,
    clients: List,
    device: str = 'cpu'
) -> Dict[int, float]:
    """
    Track ASR decay round by round after attack stops.

    Args:
        model: Global model with backdoor
        test_features: Test features for ASR
        test_labels: Test labels
        trigger_config: Trigger configuration
        n_rounds: Number of rounds to track
        server: FL server
        clients: List of honest clients only
        device: Device to run on

    Returns:
        Dictionary mapping round number to ASR
    """
    asr_history = {}

    # Track ASR each round
    for round_idx in range(n_rounds + 1):
        # Evaluate ASR at start of round
        asr = compute_attack_success_rate(
            model, test_features, test_labels, trigger_config,
            device=device
        )

        asr_history[round_idx] = asr
        print(f"Round {round_idx}: ASR = {asr:.4f}")

        # Train one round (skip after last round)
        if round_idx < n_rounds:
            def client_train_fn(client_id, global_weights):
                return clients[client_id].train(global_weights)

            server.fit_round(round_idx, client_train_fn)

    return asr_history


def compute_persistence_rate(
    asr_history: Dict[int, float],
    threshold: float = 0.8
) -> Dict[str, any]:
    """
    Compute persistence statistics.

    Args:
        asr_history: ASR over rounds
        threshold: ASR threshold for considering backdoor "active"

    Returns:
        Persistence statistics
    """
    asr_values = list(asr_history.values())
    initial_asr = asr_values[0]
    final_asr = asr_values[-1]

    # Find rounds where ASR >= threshold
    active_rounds = sum(1 for asr in asr_values if asr >= threshold)

    # Compute decay rate
    decay_rate = (initial_asr - final_asr) / max(len(asr_values) - 1, 1)

    return {
        'initial_asr': initial_asr,
        'final_asr': final_asr,
        'asr_decay': initial_asr - final_asr,
        'decay_rate_per_round': decay_rate,
        'active_rounds': active_rounds,
        'total_rounds': len(asr_values)
    }


if __name__ == "__main__":
    print("Persistence testing module loaded")
    print("Use in experiments to test backdoor durability")
