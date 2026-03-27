"""
Differential Privacy Defense Against Membership Inference Attacks

This module implements and evaluates Differential Privacy (DP) as a defense
against membership inference attacks.

Key Idea:
- Add noise to model gradients/updates to mask individual data contributions
- Higher noise (lower epsilon) → better privacy → worse attack performance

DP Mechanisms:
1. Gaussian DP: Add Gaussian noise to gradients
2. Gradient clipping: Limit gradient magnitude before adding noise
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import copy


class DPTargetTrainer:
    """
    Train target FL model with Differential Privacy.
    """

    def __init__(
        self,
        model: nn.Module,
        n_clients: int = 10,
        local_epochs: int = 5,
        client_lr: float = 0.01,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize DP-aware FL trainer.

        Args:
            model: PyTorch model to train
            n_clients: Number of FL clients
            local_epochs: Local training epochs per client
            client_lr: Learning rate
            noise_multiplier: DP noise multiplier (sigma)
            max_grad_norm: Gradient clipping norm (C)
            batch_size: Batch size
            device: Device to train on
        """
        self.model = model.to(device)
        self.n_clients = n_clients
        self.local_epochs = local_epochs
        self.client_lr = client_lr
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.device = device

        self.model_history = []

    def clip_and_add_noise(
        self,
        gradients: List[torch.Tensor],
        noise_multiplier: float,
        max_grad_norm: float
    ) -> List[torch.Tensor]:
        """
        Clip gradients and add DP noise.

        Args:
            gradients: List of gradient tensors
            noise_multiplier: Noise standard deviation
            max_grad_norm: Clipping norm

        Returns:
            Noisy gradients
        """
        clipped_gradients = []

        # Clip gradients
        total_norm = torch.sqrt(sum(g.norm()**2 for g in gradients))
        clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-10))

        for grad in gradients:
            clipped_grad = grad * clip_coef
            clipped_gradients.append(clipped_grad)

        # Add Gaussian noise
        noisy_gradients = []
        for grad in clipped_gradients:
            noise = torch.randn_like(grad) * noise_multiplier * max_grad_norm
            noisy_grad = grad + noise
            noisy_gradients.append(noisy_grad)

        return noisy_gradients

    def train_client_local_dp(
        self,
        client_model: nn.Module,
        client_data: DataLoader,
        criterion: nn.Module
    ) -> nn.Module:
        """
        Train client locally with DP.

        Args:
            client_model: Client's copy of global model
            client_data: Client's local training data
            criterion: Loss function

        Returns:
            Updated client model
        """
        client_model.train()
        optimizer = torch.optim.SGD(
            client_model.parameters(),
            lr=self.client_lr,
            momentum=0.9
        )

        for epoch in range(self.local_epochs):
            for x_batch, y_batch in client_data:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = client_model(x_batch)
                loss = criterion(outputs, y_batch)

                loss.backward()

                # Clip gradients and add noise
                gradients = [p.grad for p in client_model.parameters() if p.grad is not None]
                noisy_gradients = self.clip_and_add_noise(
                    gradients,
                    self.noise_multiplier,
                    self.max_grad_norm
                )

                # Replace gradients with noisy gradients
                for i, p in enumerate(client_model.parameters()):
                    if p.grad is not None:
                        p.grad = noisy_gradients[i]

                optimizer.step()

        return client_model

    def train_fl_model_dp(
        self,
        client_datasets: List[DataLoader],
        n_rounds: int = 20,
        verbose: bool = True
    ) -> nn.Module:
        """
        Train target FL model with DP.

        Args:
            client_datasets: List of client training datasets
            n_rounds: Number of FL communication rounds
            verbose: Print training progress

        Returns:
            Trained global model with DP
        """
        criterion = nn.CrossEntropyLoss()

        for round_idx in range(n_rounds):
            if verbose:
                print(f"Round {round_idx + 1}/{n_rounds} (DP: noise={self.noise_multiplier}, clip={self.max_grad_norm})")

            # Initialize client models from global model
            client_models = [
                type(self.model)(*(self.model.__dict__.values())).to(self.device)
                for _ in range(self.n_clients)
            ]

            global_state = self.model.state_dict()
            for client_model in client_models:
                client_model.load_state_dict(global_state)

            # Train each client locally with DP
            trained_clients = []
            client_sizes = []

            for client_model, client_data in zip(client_models, client_datasets):
                client_model = self.train_client_local_dp(client_model, client_data, criterion)
                trained_clients.append(client_model)
                client_sizes.append(len(client_data.dataset))

            # Aggregate client models (standard FedAvg)
            aggregated_state = self._federated_averaging(trained_clients, client_sizes)
            self.model.load_state_dict(aggregated_state)

            # Save model state
            self.model_history.append({
                'round': round_idx,
                'state_dict': {k: v.cpu().clone() for k, v in aggregated_state.items()}
            })

        return self.model

    def _federated_averaging(
        self,
        client_models: List[nn.Module],
        client_sizes: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client models using FedAvg."""
        global_state = client_models[0].state_dict()
        aggregated_state = {}

        total_samples = sum(client_sizes)

        for key in global_state.keys():
            aggregated_state[key] = sum(
                client_model.state_dict()[key] * client_size
                for client_model, client_size in zip(client_models, client_sizes)
            ) / total_samples

        return aggregated_state


def test_dp_defense(
    base_model_class: type,
    model_config: dict,
    client_datasets: List[DataLoader],
    member_data: DataLoader,
    nonmember_data: DataLoader,
    attack_fn,
    noise_levels: List[float],
    n_rounds: int = 10,
    n_clients: int = 10,
    device: str = 'cpu'
) -> Dict[float, Dict]:
    """
    Test DP as defense against membership inference attack.

    Args:
        base_model_class: Model class to instantiate
        model_config: Configuration dict for model
        client_datasets: Client training datasets
        member_data: Member test data
        nonmember_data: Non-member test data
        attack_fn: Attack function to test
        noise_levels: List of noise multipliers to test
        n_rounds: FL rounds to train
        n_clients: Number of clients
        device: Device to run on

    Returns:
        Dictionary of {noise_level: attack_metrics}
    """
    results = {}

    print("="*80)
    print("TESTING DIFFERENTIAL PRIVACY AS DEFENSE")
    print("="*80)

    for noise_level in noise_levels:
        print(f"\n{'='*80}")
        print(f"Testing with noise multiplier: {noise_level}")
        print(f"{'='*80}")

        # Initialize model
        model = base_model_class(**model_config)

        # Train with DP
        dp_trainer = DPTargetTrainer(
            model=model,
            n_clients=n_clients,
            local_epochs=5,
            client_lr=0.01,
            noise_multiplier=noise_level,
            max_grad_norm=1.0,
            device=device
        )

        trained_model = dp_trainer.train_fl_model_dp(
            client_datasets=client_datasets,
            n_rounds=n_rounds,
            verbose=False
        )

        # Evaluate attack
        print(f"\nEvaluating attack on DP-trained model...")
        all_scores, true_labels, _ = attack_fn(
            target_model=trained_model,
            member_data=member_data,
            nonmember_data=nonmember_data,
            device=device
        )

        # Compute metrics
        from evaluation.attack_metrics import compute_attack_metrics
        metrics = compute_attack_metrics(all_scores, true_labels)

        results[noise_level] = metrics

        print(f"\nResults (noise={noise_level}):")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

    return results


def analyze_privacy_utility_tradeoff(
    dp_results: Dict[float, Dict],
    save_path: str = None
):
    """
    Analyze and plot privacy-utility tradeoff.

    Args:
        dp_results: Results from test_dp_defense
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt

    noise_levels = sorted(dp_results.keys())
    aucs = [dp_results[nl]['auc'] for nl in noise_levels]
    accuracies = [dp_results[nl]['accuracy'] for nl in noise_levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUC vs noise
    ax1.plot(noise_levels, aucs, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Random Baseline')
    ax1.set_xlabel('Noise Multiplier (σ)', fontsize=12)
    ax1.set_ylabel('Attack AUC', fontsize=12)
    ax1.set_title('Privacy (Attack AUC) vs Noise', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy vs noise
    ax2.plot(noise_levels, accuracies, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Multiplier (σ)', fontsize=12)
    ax2.set_ylabel('Attack Accuracy', fontsize=12)
    ax2.set_title('Privacy (Attack Accuracy) vs Noise', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Privacy-utility tradeoff plot saved to {save_path}")

    plt.close()


def compute_effective_epsilon(
    noise_multiplier: float,
    n_rounds: int,
    n_clients: int,
    delta: float = 1e-5
) -> float:
    """
    Compute effective epsilon using moments accountant.

    Simplified approximation for FL with Gaussian mechanism.

    Args:
        noise_multiplier: Gaussian noise std dev
        n_rounds: Number of FL rounds
        n_clients: Number of clients per round
        delta: Delta parameter for (ε, δ)-DP

    Returns:
        Effective epsilon
    """
    # Simplified: ε ≈ q * sqrt(2 * log(1/δ)) * T / σ
    # where q = sampling rate = n_clients / total_population

    q = n_clients / 1000  # Assumed total population
    T = n_rounds

    epsilon = q * np.sqrt(2 * np.log(1.25 / delta)) * T / noise_multiplier

    return epsilon


if __name__ == "__main__":
    print("This module provides DP defense utilities.")
    print("Use via: experiments/experiment_defenses.py")
