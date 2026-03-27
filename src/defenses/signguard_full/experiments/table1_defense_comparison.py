#!/usr/bin/env python3
"""
Table 1: Defense Comparison

Generates comparison of defense accuracy under different attacks.
Compares FedAvg, Krum, TrimmedMean, FoolsGold, Bulyan, and SignGuard.
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
from datetime import datetime

from signguard import (
    SignGuardClient,
    SignGuardServer,
    SignatureManager,
    KeyStore,
    KrumDefense,
    TrimmedMeanDefense,
    FoolsGoldDefense,
    BulyanDefense,
    ModelPoisonAttack,
    LabelFlipAttack,
    BackdoorAttack,
)
from src.defenses.signguard_full.legacy.core.types import ClientConfig, ServerConfig
from torch.utils.data import DataLoader, TensorDataset


def create_synthetic_data(num_samples: int = 10000):
    """Create synthetic fraud detection dataset.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        DataLoader with synthetic data
    """
    # Generate features
    np.random.seed(42)
    X = np.random.randn(num_samples, 28).astype(np.float32)
    
    # Generate labels with some pattern
    # Fraud = 1 if sum of features > threshold
    threshold = np.percentile(X.sum(axis=1), 95)
    y = (X.sum(axis=1) > threshold).astype(np.int64)
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    return dataloader


def create_simple_model():
    """Create simple MLP model."""
    return nn.Sequential(
        nn.Linear(28, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def run_fedavg(
    dataloader: DataLoader,
    global_model: nn.Module,
    num_rounds: int = 10,
    num_clients: int = 10,
    num_byzantine: int = 2,
    attack_type: str = "none",
) -> Dict[str, float]:
    """Run FedAvg baseline.
    
    Args:
        dataloader: Training data
        global_model: Initial global model
        num_rounds: Number of FL rounds
        num_clients: Total number of clients
        num_byzantine: Number of Byzantine clients
        attack_type: Type of attack
        
    Returns:
        Metrics dictionary
    """
    from src.defenses.signguard_full.legacy.aggregation.weighted_aggregator import WeightedAggregator
    from src.defenses.signguard_full.legacy.core.types import ModelUpdate, SignedUpdate
    
    # Setup
    sm = SignatureManager()
    ks = KeyStore()
    aggregator = WeightedAggregator()
    
    # Initialize clients
    for i in range(num_clients):
        ks.generate_keypair(f"client_{i}")
    
    # Training loop
    for round_num in range(num_rounds):
        # Select clients
        client_ids = list(range(num_clients))
        
        updates = []
        for i in client_ids:
            client_id = f"client_{i}"
            
            # Apply attack if Byzantine and appropriate round
            if i < num_byzantine and attack_type != "none":
                # Create malicious update
                if attack_type == "label_flip":
                    attack = LabelFlipAttack(flip_ratio=0.3)
                    malicious_update = attack.execute(client_id, global_model.state_dict())
                    updates.append(malicious_update)
                    continue
                elif attack_type == "model_poison":
                    attack = ModelPoisonAttack(attack_type="scaling", magnitude=-5.0)
                    malicious_update = attack.execute(client_id, global_model.state_dict())
                    updates.append(malicious_update)
                    continue
            
            # Normal client (simplified - just add noise)
            update_params = {
                name: torch.randn_like(param) * 0.01
                for name, param in global_model.state_dict().items()
            }
            
            update = ModelUpdate(
                client_id=client_id,
                round_num=round_num,
                parameters=update_params,
                num_samples=100,
                metrics={"loss": 0.5, "accuracy": 0.75},
            )
            updates.append(update)
        
        # Simple average aggregation
        aggregated_params = {}
        for param_name in global_model.state_dict().keys():
            stacked = torch.stack([u.parameters[param_name] for u in updates])
            aggregated_params[param_name] = stacked.mean(dim=0)
        
        # Update global model
        global_model.load_state_dict(aggregated_params)
    
    # Evaluate
    accuracy = evaluate_model(global_model, dataloader)
    return {"accuracy": accuracy}


def run_defense(
    defense_name: str,
    dataloader: DataLoader,
    global_model: nn.Module,
    num_rounds: int = 10,
    num_clients: int = 10,
    num_byzantine: int = 2,
    attack_type: str = "none",
    cached: bool = False,
    cache_dir: str = "experiments/cache",
) -> Dict[str, float]:
    """Run a defense method.
    
    Args:
        defense_name: Name of defense method
        dataloader: Training data
        global_model: Initial global model
        num_rounds: Number of FL rounds
        num_clients: Total number of clients
        num_byzantine: Number of Byzantine clients
        attack_type: Type of attack
        cached: Whether to use cached results
        
    Returns:
        Metrics dictionary
    """
    cache_file = Path(cache_dir) / f"table1_{defense_name}_{attack_type}.json"
    
    if cached and cache_file.exists():
        print(f"  Loading cached results from {cache_file}")
        with open(cache_file) as 'r') as f:
            return json.load(f)
    
    print(f"  Running {defense_name} with {attack_type} attack...")
    
    # Setup
    sm = SignatureManager()
    ks = KeyStore()
    
    # Create defense
    if defense_name == "SignGuard":
        server = SignGuardServer(
            global_model=global_model,
            signature_manager=sm,
            config=ServerConfig(min_clients_required=5),
        )
    elif defense_name == "Krum":
        from src.defenses.signguard_full.legacy.core.types import ModelUpdate
        server = type('Server', (), {
            'aggregate': lambda updates: KrumDefense(num_byzantines=num_byzantine).aggregate(
                updates, global_model.state_dict()
            ),
        })()
    elif defense_name == "TrimmedMean":
        from src.defenses.signguard_full.legacy.core.types import ModelUpdate
        server = type('Server', (), {
            'aggregate': lambda updates: TrimmedMeanDefense(trim_ratio=0.2).aggregate(
                updates, global_model.state_dict()
            ),
        })()
    elif defense_name == "FoolsGold":
        from src.defenses.signguard_full.legacy.core.types import ModelUpdate
        server = type('Server', (), {
            'aggregate': lambda updates: FoolsGoldDefense(history_length=5).aggregate(
                updates, global_model.state_dict()
            ),
        })()
    elif defense_name == "Bulyan":
        from src.defenses.signguard_full.legacy.core.types import ModelUpdate
        server = type('Server', (), {
            'aggregate': lambda updates: BulyanDefense(num_byzantines=num_byzantine).aggregate(
                updates, global_model.state_dict()
            ),
        })()
    else:
        return run_fedavg(dataloader, global_model, num_rounds, num_clients, num_byzantine, attack_type)
    
    # Initialize clients
    server.initialize_clients([f"client_{i}" for i in range(num_clients)])
    
    # Training loop
    for round_num in range(num_rounds):
        # Select all clients
        client_ids = list(range(num_clients))
        
        signed_updates = []
        for i in client_ids:
            client_id = f"client_{i}"
            private_key = ks.get_private_key(client_id) if ks.has_client(client_id) else ks.get_private_key(f"client_{i}")
            
            # Apply attack if Byzantine
            if i < num_byzantine and attack_type != "none":
                if attack_type == "label_flip":
                    attack = LabelFlipAttack(flip_ratio=0.3)
                    malicious_update = attack.execute(client_id, global_model.state_dict())
                    signature = sm.sign_update(malicious_update, private_key)
                    public_key_str = ks.get_public_key_string(client_id)
                    from src.defenses.signguard_full.legacy.core.types import SignedUpdate
                    signed_update = SignedUpdate(
                        update=malicious_update,
                        signature=signature,
                        public_key=public_key_str,
                    )
                    signed_updates.append(signed_update)
                    continue
                elif attack_type == "backdoor":
                    attack = BackdoorAttack(trigger_pattern=torch.ones(5), target_class=1)
                    malicious_update = attack.execute(client_id, global_model.state_dict())
                    signature = sm.sign_update(malicious_update, private_key)
                    public_key_str = ks.get_public_key_string(client_id)
                    from src.defenses.signguard_full.legacy.core.types import SignedUpdate
                    signed_update = SignedUpdate(
                        update=malicious_update,
                        signature=signature,
                        public_key=public_key_str,
                    )
                    signed_updates.append(signed_update)
                    continue
                elif attack_type == "model_poison":
                    attack = ModelPoisonAttack(attack_type="scaling", magnitude=-5.0)
                    malicious_update = attack.execute(client_id, global_model.state_dict())
                    signature = sm.sign_update(malicious_update, private_key)
                    public_key_str = ks.get_public_key_string(client_id)
                    from src.defenses.signguard_full.legacy.core.types import SignedUpdate
                    signed_update = SignedUpdate(
                        update=malicious_update,
                        signature=signature,
                        public_key=public_key_str,
                    )
                    signed_updates.append(signed_update)
                    continue
            
            # Normal client
            update_params = {
                name: torch.randn_like(param) * 0.01
                for name, param in global_model.state_dict().items()
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
        if defense_name == "SignGuard":
            result = server.aggregate(signed_updates)
            global_model.load_state_dict(result.global_model)
        elif hasattr(server, 'aggregate'):
            from src.defenses.signguard_full.legacy.core.types import ModelUpdate
            # Convert to ModelUpdate
            updates = [su.update for su in signed_updates]
            result = server.aggregate(updates)
            global_model.load_state_dict(result.global_model)
    
    # Evaluate
    accuracy = evaluate_model(global_model, dataloader)
    metrics = {"accuracy": accuracy}
    
    # Cache results
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> float:
    """Evaluate model accuracy.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        
    Returns:
        Accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return correct / total if total > 0 else 0.0


def main():
    """Run Table 1 experiment: Defense comparison."""
    print("=" * 60)
    print("Table 1: Defense Comparison")
    print("=" * 60)
    print()
    
    # Configuration
    defenses = ["FedAvg", "Krum", "TrimmedMean", "FoolsGold", "Bulyan", "SignGuard"]
    attack_types = ["none", "label_flip", "backdoor", "model_poison"]
    
    num_rounds = 10
    num_clients = 10
    num_byzantine = 2
    use_cache = True
    
    # Create data
    print("Creating synthetic dataset...")
    dataloader = create_synthetic_data(num_samples=5000)
    print(f"  Created dataset with {len(dataloader.dataset)} samples")
    print()
    
    # Run experiments
    results = {defense: {} for defense in defenses}
    
    for defense in defenses:
        print(f"Testing {defense}:")
        results[defense] = {}
        
        for attack in attack_types:
            # Create fresh model for each experiment
            model = create_simple_model()
            
            try:
                metrics = run_defense(
                    defense_name=defense,
                    dataloader=dataloader,
                    global_model=model,
                    num_rounds=num_rounds,
                    num_clients=num_clients,
                    num_byzantine=num_byzantine,
                    attack_type=attack,
                    cached=use_cache,
                )
                accuracy = metrics["accuracy"]
                results[defense][attack] = accuracy
                print(f"    {attack}: {accuracy:.3f}")
            except Exception as e:
                print(f"    {attack}: Error - {e}")
                results[defense][attack] = 0.0
        
        print()
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "table1_defense_comparison.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 60)
    print("Results saved to:", results_file)
    print("=" * 60)
    
    # Print table
    print("\nTable 1: Defense Accuracy Comparison")
    print("-" * 60)
    header = f"{'Defense':<15} {'No Attack':<12} {'Label Flip':<12} {'Backdoor':<12} {'Model Poison':<12}"
    print(header)
    print("-" * 60)
    
    for defense in defenses:
        row = f"{defense:<15}"
        for attack in attack_types[1:]:  # Skip "none"
            acc = results[defense].get(attack, 0.0)
            row += f" {acc:<11.3f}"
        print(row)
    
    return results


if __name__ == "__main__":
    main()
