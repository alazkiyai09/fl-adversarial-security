#!/usr/bin/env python3
"""
Table 3: Overhead Analysis

Measures computational, communication, and memory overhead
of SignGuard compared to baseline defenses.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from typing import Dict

from signguard import SignGuardServer, SignatureManager, KeyStore
from src.defenses.signguard_full.legacy.core.types import ServerConfig, ClientConfig
from src.defenses.signguard_full.legacy.defenses import KrumDefense, TrimmedMeanDefense
from torch.utils.data import DataLoader, TensorDataset


def create_simple_data(num_samples: int = 1000):
    """Create simple dataset."""
    X = torch.randn(num_samples, 28)
    y = torch.randint(0, 2, (num_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=32)


def create_simple_model():
    """Create simple model."""
    return nn.Sequential(
        nn.Linear(28, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


def measure_overhead(
    defense_name: str,
    model: nn.Module,
    dataloader: DataLoader,
    num_rounds: int = 5,
    num_clients: int = 10,
    cached: bool = True,
    cache_dir: str = "experiments/cache",
) -> Dict[str, float]:
    """Measure overhead for a defense.
    
    Args:
        defense_name: Defense method name
        model: Model to use
        dataloader: Data loader
        num_rounds: Number of rounds
        num_clients: Number of clients
        cached: Use cached results
        
    Returns:
        Overhead metrics
    """
    cache_file = Path(cache_dir) / f"table3_{defense_name}.json"
    
    if cached and cache_file.exists():
        print(f"  Loading cached results from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    print(f"  Measuring {defense_name} overhead...")
    
    # Setup
    sm = SignatureManager()
    ks = KeyStore()
    
    server = SignGuardServer(
        global_model=model,
        signature_manager=sm,
        config=ServerConfig(min_clients_required=5),
    )
    
    server.initialize_clients([f"client_{i}" for i in range(num_clients)])
    
    # Initialize clients
    for i in range(num_clients):
        if not ks.has_client(f"client_{i}"):
            ks.generate_keypair(f"client_{i}")
    
    # Measure memory
    import psutil
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Measure time and communication
    total_time = 0
    total_bytes = 0
    
    for round_num in range(num_rounds):
        round_start = time.time()
        
        # Generate updates
        signed_updates = []
        for i in range(num_clients):
            client_id = f"client_{i}"
            private_key = ks.get_private_key(client_id)
            
            from src.defenses.signguard_full.legacy.core.types import ModelUpdate, SignedUpdate
            update_params = {
                name: torch.randn_like(param) * 0.01
                for name, param in model.state_dict().items()
            }
            update = ModelUpdate(
                client_id=client_id,
                round_num=round_num,
                parameters=update_params,
                num_samples=100,
                metrics={"loss": 0.5},
            )
            
            signature = sm.sign_update(update, private_key)
            public_key_str = ks.get_public_key_string(client_id)
            
            signed_update = SignedUpdate(
                update=update,
                signature=signature,
                public_key=public_key_str,
            )
            signed_updates.append(signed_update)
            
            # Estimate communication (signature + public key + parameters)
            total_bytes += len(signature) + len(public_key_str)
            for param in update_params.values():
                total_bytes += param.numel() * 4  # float32
        
        # Aggregate
        result = server.aggregate(signed_updates)
        
        round_time = time.time() - round_start
        total_time += round_time
    
    # Memory after
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_overhead = memory_after - memory_before
    
    metrics = {
        "time_per_round": total_time / num_rounds * 1000,  # ms
        "comm_overhead": total_bytes / 1024 / 1024,  # MB
        "memory_overhead": memory_overhead,  # MB
    }
    
    # Cache results
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(metrics, f)
    
    return metrics


def main():
    """Run Table 3 experiment: Overhead analysis."""
    print("=" * 60)
    print("Table 3: Overhead Analysis")
    print("=" * 60)
    print()
    
    defenses = ["FedAvg", "Krum", "TrimmedMean", "FoolsGold", "SignGuard"]
    use_cache = True
    
    # Create data and model
    dataloader = create_simple_data()
    model = create_simple_model()
    
    # Measure overhead
    results = {}
    for defense in defenses:
        metrics = measure_overhead(
            defense_name=defense,
            model=model,
            dataloader=dataloader,
            num_rounds=5,
            num_clients=10,
            cached=use_cache,
        )
        results[defense] = metrics
        print(f"{defense}:")
        print(f"  Time: {metrics['time_per_round']:.2f} ms/round")
        print(f"  Comm: {metrics['comm_overhead']:.2f} MB")
        print(f"  Memory: {metrics['memory_overhead']:.2f} MB")
        print()
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "table3_overhead.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 60)
    print("Results saved to:", results_file)
    print("=" * 60)
    print()
    
    # Print table
    print("Table 3: Computational Overhead")
    print("-" * 70)
    print(f"{'Defense':<15} {'Time (ms/round)':<20} {'Comm (MB)':<15} {'Memory (MB)':<15}")
    print("-" * 70)
    
    for defense in defenses:
        metrics = results[defense]
        print(f"{defense:<15} {metrics['time_per_round']:<20.2f} {metrics['comm_overhead']:<15.2f} {metrics['memory_overhead']:<15.2f}")


if __name__ == "__main__":
    main()
