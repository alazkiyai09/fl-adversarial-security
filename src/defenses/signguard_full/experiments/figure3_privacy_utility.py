#!/usr/bin/env python3
"""
Figure 3: Privacy-Utility Trade-off

Plots the trade-off between privacy (DP epsilon) and utility (accuracy/ASR)
when combining SignGuard with differential privacy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

from signguard import SignGuardServer, SignatureManager, KeyStore
from src.defenses.signguard_full.legacy.core.types import ServerConfig
from src.defenses.signguard_full.legacy.attacks import BackdoorAttack
from src.defenses.signguard_full.legacy.utils.visualization import plot_privacy_utility
from src.defenses.signguard_full.legacy.utils.metrics import compute_accuracy, compute_attack_success_rate
from torch.utils.data import DataLoader, TensorDataset


def create_simple_data(num_samples: int = 2000):
    """Create dataset."""
    X = torch.randn(num_samples, 28)
    y = torch.randint(0, 2, (num_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=32)


def create_simple_model():
    """Create model."""
    return nn.Sequential(
        nn.Linear(28, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def add_dp_noise(
    parameters: dict[str, torch.Tensor],
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Add Gaussian noise for DP-SGD.
    
    Args:
        parameters: Model parameters
        epsilon: Privacy budget
        delta: Delta parameter
        sensitivity: Sensitivity of gradients
        
    Returns:
        Noisy parameters
    """
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    
    noisy_params = {}
    for name, param in parameters.items():
        noise = torch.randn_like(param) * sigma
        noisy_params[name] = param + noise
    
    return noisy_params


def simulate_dp_fl(
    epsilon: float,
    dataloader: DataLoader,
    model: nn.Module,
    num_rounds: int = 10,
    num_clients: int = 10,
    use_dp: bool = True,
) -> tuple:
    """Simulate FL with DP.
    
    Args:
        epsilon: Privacy budget
        dataloader: Training data
        model: Model
        num_rounds: Number of rounds
        num_clients: Number of clients
        use_dp: Whether to use DP
        
    Returns:
        Tuple of (accuracy, asr)
    """
    # Setup
    sm = SignatureManager()
    ks = KeyStore()
    
    server = SignGuardServer(
        global_model=model,
        signature_manager=sm,
        config=ServerConfig(min_clients_required=5),
    )
    
    server.initialize_clients([f"client_{i}" for i in range(num_clients)])
    
    # FL rounds
    for round_num in range(num_rounds):
        signed_updates = []
        
        for i in range(num_clients):
            client_id = f"client_{i}"
            
            # Generate update
            update_params = {
                name: torch.randn_like(param) * 0.01
                for name, param in model.state_dict().items()
            }
            
            # Add DP noise if enabled
            if use_dp:
                update_params = add_dp_noise(update_params, epsilon)
            
            from src.defenses.signguard_full.legacy.core.types import ModelUpdate, SignedUpdate
            update = ModelUpdate(
                client_id=client_id,
                round_num=round_num,
                parameters=update_params,
                num_samples=100,
                metrics={"loss": 0.5},
            )
            
            if not ks.has_client(client_id):
                ks.generate_keypair(client_id)
            private_key = ks.get_private_key(client_id)
            
            signature = sm.sign_update(update, private_key)
            public_key_str = ks.get_public_key_string(client_id)
            
            signed_update = SignedUpdate(
                update=update,
                signature=signature,
                public_key=public_key_str,
            )
            signed_updates.append(signed_update)
        
        # Aggregate
        result = server.aggregate(signed_updates)
        model.load_state_dict(result.global_model)
    
    # Evaluate accuracy
    accuracy = compute_accuracy(model, dataloader)
    
    # Evaluate ASR (backdoor attack success)
    trigger_pattern = torch.ones(5)
    asr = compute_attack_success_rate(
        model=model,
        test_loader=dataloader,
        target_class=1,
        trigger_pattern=trigger_pattern,
    )
    
    return accuracy, asr


def run_privacy_utility_experiment(
    epsilon_values: list,
    cached: bool = True,
    cache_dir: str = "experiments/cache",
):
    """Run privacy-utility trade-off experiment.
    
    Args:
        epsilon_values: List of epsilon values to test
        cached: Use cached results
        
    Returns:
        Results dictionary
    """
    cache_file = Path(cache_dir) / "figure3_privacy_utility.json"
    
    if cached and cache_file.exists():
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    print("Running privacy-utility experiment...")
    
    # Create data
    dataloader = create_simple_data()
    
    results = {
        "epsilon_values": epsilon_values,
        "accuracy_values": {"SignGuard": [], "SignGuard+DP": []},
        "asr_values": {"SignGuard": [], "SignGuard+DP": []},
    }
    
    for eps in epsilon_values:
        print(f"  Testing epsilon = {eps}")
        
        # SignGuard without DP
        model_no_dp = create_simple_model()
        acc_no_dp, asr_no_dp = simulate_dp_fl(
            epsilon=eps,
            dataloader=dataloader,
            model=model_no_dp,
            num_rounds=10,
            num_clients=10,
            use_dp=False,
        )
        results["accuracy_values"]["SignGuard"].append(acc_no_dp)
        results["asr_values"]["SignGuard"].append(asr_no_dp)
        
        # SignGuard with DP
        model_with_dp = create_simple_model()
        acc_dp, asr_dp = simulate_dp_fl(
            epsilon=eps,
            dataloader=dataloader,
            model=model_with_dp,
            num_rounds=10,
            num_clients=10,
            use_dp=True,
        )
        results["accuracy_values"]["SignGuard+DP"].append(acc_dp)
        results["asr_values"]["SignGuard+DP"].append(asr_dp)
    
    # Cache results
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(results, f)
    
    return results


def main():
    """Generate Figure 3: Privacy-Utility trade-off."""
    print("=" * 60)
    print("Figure 3: Privacy-Utility Trade-off")
    print("=" * 60)
    print()
    
    epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    
    # Run experiment
    results = run_privacy_utility_experiment(
        epsilon_values=epsilon_values,
        cached=True,
    )
    
    # Print results
    print()
    print("Results:")
    print("-" * 60)
    print(f"{'Epsilon':<10} {'Acc (SG)':<12} {'Acc (DP)':<12} {'ASR (SG)':<12} {'ASR (DP)':<12}")
    print("-" * 60)
    
    for i, eps in enumerate(epsilon_values):
        acc_sg = results["accuracy_values"]["SignGuard"][i]
        acc_dp = results["accuracy_values"]["SignGuard+DP"][i]
        asr_sg = results["asr_values"]["SignGuard"][i]
        asr_dp = results["asr_values"]["SignGuard+DP"][i]
        
        print(f"{eps:<10.1f} {acc_sg:<12.3f} {acc_dp:<12.3f} {asr_sg:<12.3f} {asr_dp:<12.3f}")
    
    # Check matplotlib
    try:
        import matplotlib.pyplot as plt
        has_mpl = True
    except ImportError:
        has_mpl = False
    
    if has_mpl:
        # Generate figure
        fig = plot_privacy_utility(
            epsilon_values=results["epsilon_values"],
            accuracy_values=results["accuracy_values"],
            asr_values=results["asr_values"],
            output_path="figures/plots/figure3_privacy_utility.pdf",
        )
        plt.close(fig)
        print()
        print("Figure 3 saved to: figures/plots/figure3_privacy_utility.pdf")
    else:
        print()
        print("Note: Matplotlib not available. Saving data for offline plotting.")
        output_dir = Path("figures/data")
        output_dir.mkdir(parents=True, exist_ok=True)
        data_file = output_dir / "figure3_privacy_data.json"
        with open(data_file, 'w') as f:
            json.dump(results, f)
        print(f"Data saved to: {data_file}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
