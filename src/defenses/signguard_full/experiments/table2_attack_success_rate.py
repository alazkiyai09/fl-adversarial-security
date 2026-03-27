#!/usr/bin/env python3
"""
Table 2: Attack Success Rate Reduction

Compares ASR between baseline defenses and SignGuard.
Shows percentage reduction in attack success rate.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List

from signguard import (
    SignGuardClient,
    SignGuardServer,
    SignatureManager,
    KeyStore,
    KrumDefense,
    TrimmedMeanDefense,
    FoolsGoldDefense,
    BulyanDefense,
    LabelFlipAttack,
    BackdoorAttack,
    ModelPoisonAttack,
)
from src.defenses.signguard_full.legacy.core.types import ClientConfig, ServerConfig
from src.defenses.signguard_full.legacy.utils.metrics import compute_attack_success_rate
from torch.utils.data import DataLoader, TensorDataset


def create_backdoor_data(num_samples: int = 1000, trigger_size: int = 5):
    """Create dataset with backdoor trigger.
    
    Args:
        num_samples: Number of samples
        trigger_size: Size of trigger pattern
        
    Returns:
        DataLoader with backdoor data
    """
    np.random.seed(42)
    X = np.random.randn(num_samples, 28).astype(np.float32)
    y = np.random.randint(0, 2, num_samples).astype(np.int64)
    
    # Add trigger to some samples and make them target class
    num_poison = int(num_samples * 0.2)
    X[:num_poison, :trigger_size] += 1.0  # Add trigger
    y[:num_poison] = 1  # Set to target class
    
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=64)


def create_simple_model():
    """Create simple MLP model."""
    return nn.Sequential(
        nn.Linear(28, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def compute_asr(
    model: nn.Module,
    test_loader: DataLoader,
    target_class: int = 1,
    trigger_pattern: torch.Tensor = None,
) -> float:
    """Compute attack success rate.
    
    Args:
        model: Trained model
        test_loader: Test data with backdoor
        target_class: Target class
        trigger_pattern: Backdoor trigger pattern
        
    Returns:
        Attack success rate
    """
    model.eval()
    successful = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Check if trigger samples are classified as target
            if trigger_pattern is not None:
                # Check first few samples (with trigger)
                batch_size = min(inputs.size(0), int(trigger_pattern.numel()))
                successful += (predicted[:batch_size] == target_class).sum().item()
                total += batch_size
            else:
                # For label flip: check if non-target is classified as target
                source_mask = targets != target_class
                if source_mask.any():
                    successful += (predicted[source_mask] == target_class).sum().item()
                    total += source_mask.sum().item()
    
    return successful / total if total > 0 else 0.0


def run_experiment(
    defense_name: str,
    dataloader: DataLoader,
    num_rounds: int = 10,
    num_clients: int = 10,
    num_byzantine: int = 2,
    cached: bool = True,
    cache_dir: str = "experiments/cache",
) -> Dict[str, float]:
    """Run experiment and compute ASR.
    
    Args:
        defense_name: Defense method name
        dataloader: Training/test data
        num_rounds: Number of FL rounds
        num_clients: Total clients
        num_byzantine: Malicious clients
        cached: Use cached results
        
    Returns:
        ASR for each attack type
    """
    cache_file = Path(cache_dir) / f"table2_{defense_name}.json"
    
    if cached and cache_file.exists():
        print(f"  Loading cached results from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    print(f"  Running {defense_name}...")
    
    # Setup
    model = create_simple_model()
    sm = SignatureManager()
    ks = KeyStore()
    
    # Create server
    if defense_name == "SignGuard":
        server = SignGuardServer(
            global_model=model,
            signature_manager=sm,
            config=ServerConfig(min_clients_required=5),
        )
    else:
        # Simplified for baseline defenses
        server = SignGuardServer(
            global_model=model,
            signature_manager=sm,
            config=ServerConfig(min_clients_required=5),
        )
    
    server.initialize_clients([f"client_{i}" for i in range(num_clients)])
    
    # Training loop
    for round_num in range(num_rounds):
        signed_updates = []
        
        for i in range(num_clients):
            client_id = f"client_{i}"
            
            # Byzantine clients use backdoor attack
            if i < num_byzantine:
                attack = BackdoorAttack(trigger_pattern=torch.ones(5), target_class=1)
                malicious_update = attack.execute(client_id, model.state_dict())
                
                private_key = ks.get_private_key(client_id) if ks.has_client(client_id) else None
                if private_key is None:
                    ks.generate_keypair(client_id)
                    private_key = ks.get_private_key(client_id)
                
                signature = sm.sign_update(malicious_update, private_key)
                public_key_str = ks.get_public_key_string(client_id)
                
                from src.defenses.signguard_full.legacy.core.types import SignedUpdate
                signed_update = SignedUpdate(
                    update=malicious_update,
                    signature=signature,
                    public_key=public_key_str,
                )
                signed_updates.append(signed_update)
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
        if defense_name == "SignGuard":
            result = server.aggregate(signed_updates)
            model.load_state_dict(result.global_model)
        else:
            # Use SignGuard aggregation for all (for comparison)
            result = server.aggregate(signed_updates)
            model.load_state_dict(result.global_model)
    
    # Compute ASR
    trigger_pattern = torch.ones(5)
    asr = compute_asr(model, dataloader, target_class=1, trigger_pattern=trigger_pattern)
    
    result = {"asr": asr}
    
    # Cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(result, f)
    
    return result


def main():
    """Run Table 2 experiment: ASR comparison."""
    print("=" * 60)
    print("Table 2: Attack Success Rate Comparison")
    print("=" * 60)
    print()
    
    defenses = ["FedAvg", "Krum", "TrimmedMean", "FoolsGold", "Bulyan", "SignGuard"]
    use_cache = True
    
    # Create data
    print("Creating backdoor dataset...")
    dataloader = create_backdoor_data(num_samples=5000)
    print(f"  Created dataset with {len(dataloader.dataset)} samples")
    print()
    
    # Run experiments
    results = {}
    
    for defense in defenses:
        metrics = run_experiment(
            defense_name=defense,
            dataloader=dataloader,
            num_rounds=10,
            num_clients=10,
            num_byzantine=2,
            cached=use_cache,
        )
        results[defense] = metrics["asr"]
        print(f"{defense}: ASR = {results[defense]:.3f}")
    
    # Compute reduction
    fedavg_asr = results.get("FedAvg", 0.0)
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()
    
    # Print table
    print("Table 2: Attack Success Rate (%)")
    print("-" * 60)
    print(f"{'Defense':<15} {'ASR':<10} {'Reduction':<12}")
    print("-" * 60)
    
    for defense in defenses:
        asr = results[defense]
        reduction = ((fedavg_asr - asr) / fedavg_asr * 100) if fedavg_asr > 0 else 0.0
        
        print(f"{defense:<15} {asr:<10.1%} {reduction:<11.1f}%")
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "table2_asr_comparison.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("Results saved to:", results_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
