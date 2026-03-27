#!/usr/bin/env python3
"""
Figure 1: Reputation Evolution

Plots reputation trajectories of honest and malicious clients
over federated learning rounds.
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
from src.defenses.signguard_full.legacy.attacks import ModelPoisonAttack
from src.defenses.signguard_full.legacy.utils.visualization import plot_reputation_evolution
from torch.utils.data import DataLoader, TensorDataset


def create_simple_data(num_samples: int = 1000):
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


def simulate_fl_with_reputation_tracking(
    num_rounds: int = 20,
    num_clients: int = 10,
    num_byzantine: int = 2,
    cached: bool = True,
    cache_dir: str = "experiments/cache",
) -> tuple:
    """Simulate FL and track reputation evolution.
    
    Args:
        num_rounds: Number of FL rounds
        num_clients: Total clients
        num_byzantine: Malicious clients
        cached: Use cached results
        
    Returns:
        Tuple of (reputation_history, honest_ids, malicious_ids)
    """
    cache_file = Path(cache_dir) / "figure1_reputation_history.json"
    
    if cached and cache_file.exists():
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return (
            data["reputation_history"],
            data["honest_clients"],
            data["malicious_clients"],
        )
    
    print("Simulating FL with reputation tracking...")
    
    # Setup
    model = create_simple_model()
    dataloader = create_simple_data()
    sm = SignatureManager()
    ks = KeyStore()
    
    server = SignGuardServer(
        global_model=model,
        signature_manager=sm,
        config=ServerConfig(min_clients_required=5),
    )
    
    client_ids = [f"client_{i}" for i in range(num_clients)]
    server.initialize_clients(client_ids)
    
    # Track reputation
    reputation_history = {cid: [] for cid in client_ids}
    
    honest_ids = client_ids[num_byzantine:]
    malicious_ids = client_ids[:num_byzantine]
    
    # FL rounds
    for round_num in range(num_rounds):
        # Generate updates
        signed_updates = []
        
        for i, client_id in enumerate(client_ids):
            private_key = ks.get_private_key(client_id) if ks.has_client(client_id) else None
            if private_key is None:
                ks.generate_keypair(client_id)
                private_key = ks.get_private_key(client_id)
            
            # Byzantine attack
            if i < num_byzantine:
                attack = ModelPoisonAttack(attack_type="scaling", magnitude=-5.0)
                malicious_update = attack.execute(client_id, model.state_dict())
                
                signature = sm.sign_update(malicious_update, private_key)
                public_key_str = ks.get_public_key_string(client_id)
                
                from src.defenses.signguard_full.legacy.core.types import SignedUpdate
                signed_update = SignedUpdate(
                    update=malicious_update,
                    signature=signature,
                    public_key=public_key_str,
                )
            else:
                # Honest client
                update_params = {
                    name: torch.randn_like(param) * 0.01
                    for name, param in model.state_dict().items()
                }
                
                from src.defenses.signguard_full.legacy.core.types import ModelUpdate, SignedUpdate
                update = ModelUpdate(
                    client_id=client_id,
                    round_num=round_num,
                    parameters=update_params,
                    num_samples=100,
                    metrics={"loss": 0.5, "accuracy": 0.75},
                )
                
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
        
        # Track reputations
        reputations = server.get_reputations()
        for cid, rep in reputations.items():
            reputation_history[cid].append(rep)
    
    # Cache results
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "reputation_history": reputation_history,
        "honest_clients": honest_ids,
        "malicious_clients": malicious_ids,
    }
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return reputation_history, honest_ids, malicious_ids


def main():
    """Generate Figure 1: Reputation evolution."""
    print("=" * 60)
    print("Figure 1: Reputation Evolution")
    print("=" * 60)
    print()
    
    # Simulate FL
    reputation_history, honest_ids, malicious_ids = simulate_fl_with_reputation_tracking(
        num_rounds=20,
        num_clients=10,
        num_byzantine=2,
        cached=True,
    )
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        has_mpl = True
    except ImportError:
        has_mpl = False
        print("Note: Matplotlib not available. Saving data for offline plotting.")
        print("Install matplotlib with: pip install matplotlib")
        print()
    
    if has_mpl:
        # Generate figure
        fig = plot_reputation_evolution(
            reputation_history=reputation_history,
            honest_clients=honest_ids,
            malicious_clients=malicious_ids,
            output_path="figures/plots/figure1_reputation_evolution.pdf",
        )
        plt.close(fig)
        print("Figure 1 saved to: figures/plots/figure1_reputation_evolution.pdf")
    else:
        # Save data for offline plotting
        output_dir = Path("figures/data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = output_dir / "figure1_reputation_data.json"
        with open(data_file, 'w') as f:
            json.dump(reputation_history, f)
        
        print(f"Reputation data saved to: {data_file}")
        print("Install matplotlib to generate figure automatically")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
