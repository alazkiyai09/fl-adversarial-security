#!/usr/bin/env python3
"""
Ablation Study: Component Contribution

Evaluates contribution of each SignGuard component:
- Crypto only
- Detection only
- Reputation only
- Crypto + Detection
- Crypto + Reputation
- Full SignGuard (all components)
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
from src.defenses.signguard_full.legacy.detection import EnsembleDetector
from src.defenses.signguard_full.legacy.reputation import DecayReputationSystem
from src.defenses.signguard_full.legacy.aggregation import WeightedAggregator
from src.defenses.signguard_full.legacy.core.types import ServerConfig
from src.defenses.signguard_full.legacy.attacks import ModelPoisonAttack
from src.defenses.signguard_full.legacy.utils.visualization import plot_ablation_study
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


class CustomServer:
    """Custom server for ablation study."""
    
    def __init__(
        self,
        global_model: nn.Module,
        use_crypto: bool = True,
        use_detection: bool = True,
        use_reputation: bool = True,
    ):
        self.global_model = global_model
        self.use_crypto = use_crypto
        self.use_detection = use_detection
        self.use_reputation = use_reputation
        
        if use_crypto:
            self.signature_manager = SignatureManager()
            self.key_store = KeyStore()
        
        if use_detection:
            self.detector = EnsembleDetector()
        
        if use_reputation:
            self.reputation_system = DecayReputationSystem()
        
        self.aggregator = WeightedAggregator()
        self.current_round = 0
    
    def aggregate(self, updates, global_params):
        """Aggregate updates with selected components."""
        # Filter by crypto
        if self.use_crypto:
            verified_updates = []
            for update in updates:
                if hasattr(update, 'signature'):
                    if self.signature_manager.verify_update(update):
                        verified_updates.append(update)
                else:
                    verified_updates.append(update)
        else:
            verified_updates = updates
        
        # Detection
        if self.use_detection:
            self.detector.update_statistics(verified_updates, global_params)
        
        # Reputation
        reputations = {}
        if self.use_reputation:
            for update in verified_updates:
                client_id = update.client_id if hasattr(update, 'client_id') else "unknown"
                if client_id not in self.reputation_system.reputations:
                    self.reputation_system.initialize_client(client_id)
                
                if self.use_detection and hasattr(update, 'metrics'):
                    anomaly_score = self.detector.compute_anomaly_score(update, global_params)
                    self.reputation_system.update_reputation(
                        client_id, anomaly_score.combined_score, self.current_round
                    )
                else:
                    self.reputation_system.update_reputation(client_id, 0.0, self.current_round)
                
                reputations[client_id] = self.reputation_system.get_reputation(client_id)
        else:
            for update in verified_updates:
                client_id = update.client_id if hasattr(update, 'client_id') else "unknown"
                reputations[client_id] = 1.0
        
        # Aggregate
        from src.defenses.signguard_full.legacy.core.types import AggregationResult
        aggregated_params = {}
        for param_name in global_params.keys():
            weighted_sum = torch.zeros_like(global_params[param_name])
            for update in verified_updates:
                client_id = update.client_id if hasattr(update, 'client_id') else "unknown"
                weight = reputations.get(client_id, 1.0)
                weighted_sum += weight * update.parameters[param_name]
            
            aggregated_params[param_name] = weighted_sum / len(verified_updates)
        
        return aggregated_params


def run_configuration(
    config_name: str,
    use_crypto: bool,
    use_detection: bool,
    use_reputation: bool,
    dataloader: DataLoader,
    model: nn.Module,
    num_rounds: int = 10,
    num_clients: int = 10,
    num_byzantine: int = 2,
    cached: bool = True,
    cache_dir: str = "experiments/cache",
) -> tuple:
    """Run a specific configuration.
    
    Returns:
        Tuple of (accuracy, asr)
    """
    cache_file = Path(cache_dir) / f"ablation_{config_name}.json"
    
    if cached and cache_file.exists():
        print(f"  Loading cached results for {config_name}")
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return data["accuracy"], data["asr"]
    
    print(f"  Running {config_name}...")
    
    # Setup
    server = CustomServer(
        global_model=model,
        use_crypto=use_crypto,
        use_detection=use_detection,
        use_reputation=use_reputation,
    )
    
    # Initialize keys if using crypto
    if use_crypto:
        for i in range(num_clients):
            server.key_store.generate_keypair(f"client_{i}")
    
    # FL rounds
    for round_num in range(num_rounds):
        updates = []
        
        for i in range(num_clients):
            client_id = f"client_{i}"
            
            # Byzantine attack
            if i < num_byzantine:
                attack = ModelPoisonAttack(attack_type="scaling", magnitude=-5.0)
                malicious_update = attack.execute(client_id, model.state_dict())
                updates.append(malicious_update)
            else:
                # Honest update
                update_params = {
                    name: torch.randn_like(param) * 0.01
                    for name, param in model.state_dict().items()
                }
                
                from src.defenses.signguard_full.legacy.core.types import ModelUpdate
                updates.append(ModelUpdate(
                    client_id=client_id,
                    round_num=round_num,
                    parameters=update_params,
                    num_samples=100,
                    metrics={"loss": 0.5},
                ))
        
        # Aggregate
        new_params = server.aggregate(updates, model.state_dict())
        model.load_state_dict(new_params)
    
    # Evaluate
    accuracy = compute_accuracy(model, dataloader)
    
    # ASR (model poison success)
    trigger_pattern = torch.ones(5)
    asr = compute_attack_success_rate(
        model=model,
        test_loader=dataloader,
        target_class=1,
        trigger_pattern=trigger_pattern,
    )
    
    # Cache results
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump({"accuracy": accuracy, "asr": asr}, f)
    
    return accuracy, asr


def main():
    """Run ablation study."""
    print("=" * 60)
    print("Ablation Study: Component Contribution")
    print("=" * 60)
    print()
    
    # Configurations
    configs = [
        ("No Defense", False, False, False),
        ("Crypto Only", True, False, False),
        ("Detection Only", False, True, False),
        ("Reputation Only", False, False, True),
        ("Crypto + Detection", True, True, False),
        ("Crypto + Reputation", True, False, True),
        ("SignGuard (All)", True, True, True),
    ]
    
    # Create data and model
    dataloader = create_simple_data()
    
    results = {
        "component_names": [],
        "accuracy_values": [],
        "asr_values": [],
    }
    
    use_cache = True
    
    for config_name, use_crypto, use_detection, use_reputation in configs:
        model = create_simple_model()
        
        accuracy, asr = run_configuration(
            config_name=config_name,
            use_crypto=use_crypto,
            use_detection=use_detection,
            use_reputation=use_reputation,
            dataloader=dataloader,
            model=model,
            num_rounds=10,
            num_clients=10,
            num_byzantine=2,
            cached=use_cache,
        )
        
        results["component_names"].append(config_name)
        results["accuracy_values"].append(accuracy)
        results["asr_values"].append(asr)
        
        print(f"    Accuracy: {accuracy:.3f}, ASR: {asr:.3f}")
    
    print()
    print("=" * 60)
    print("Ablation Study Results:")
    print("-" * 60)
    print(f"{'Configuration':<20} {'Accuracy':<12} {'ASR':<10}")
    print("-" * 60)
    
    for i, name in enumerate(results["component_names"]):
        acc = results["accuracy_values"][i]
        asr = results["asr_values"][i]
        print(f"{name:<20} {acc:<12.3f} {asr:<10.3f}")
    
    # Check matplotlib
    try:
        import matplotlib.pyplot as plt
        has_mpl = True
    except ImportError:
        has_mpl = False
    
    if has_mpl:
        fig = plot_ablation_study(
            component_names=results["component_names"],
            accuracy_values=results["accuracy_values"],
            asr_values=results["asr_values"],
            output_path="figures/plots/ablation_study.pdf",
        )
        plt.close(fig)
        print()
        print("Ablation figure saved to: figures/plots/ablation_study.pdf")
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "ablation_study.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to:", results_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
