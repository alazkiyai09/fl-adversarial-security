"""
Main experiment script for backdoor attack on federated learning.
Runs FL simulation with backdoor attack and measures persistence.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

from src.attacks.backdoor.legacy.models.fraud_model import FraudMLP
from src.attacks.backdoor.legacy.utils.data_loader import (
    generate_fraud_data, partition_data_iid, create_dataloaders, load_config
)
from src.attacks.backdoor.legacy.clients.honest_client import HonestClient
from src.attacks.backdoor.legacy.clients.malicious_client import MaliciousClient
from src.attacks.backdoor.legacy.servers.fl_server import FlowerFLServer
from src.attacks.backdoor.legacy.metrics.attack_metrics import evaluate_backdoor_attack
from src.attacks.backdoor.legacy.metrics.persistence import test_backdoor_persistence, track_asr_over_rounds


class BackdoorExperiment:
    """
    End-to-end backdoor attack experiment on federated learning.
    """

    def __init__(
        self,
        data_config_path: str = 'config/data.yaml',
        attack_config_path: str = 'config/attack.yaml',
        results_dir: str = 'results',
        device: str = 'cpu'
    ):
        """
        Initialize experiment.

        Args:
            data_config_path: Path to data config
            attack_config_path: Path to attack config
            results_dir: Directory to save results
            device: Device to run on
        """
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Load configs
        self.data_config = load_config(data_config_path)
        self.attack_config = load_config(attack_config_path)

        # Extract configs
        self.config = self.attack_config.get('training', {})
        self.fed_config = self.attack_config.get('federated', {})

        # Training parameters
        self.num_clients = self.data_config['data']['num_clients']
        self.samples_per_client = self.data_config['data']['samples_per_client']
        self.num_rounds = self.fed_config.get('num_rounds', 70)
        self.attack_duration = self.attack_config['attack'].get('attack_duration', 50)

        # Attack parameters
        self.num_malicious = self.attack_config['attack'].get('num_malicious', 1)
        self.malicious_ids = self.attack_config['attack'].get('malicious_client_ids', [0])

        # Results storage
        self.history = {
            'round': [],
            'clean_accuracy': [],
            'attack_success_rate': [],
            'class_0_accuracy': [],
            'class_1_accuracy': []
        }

    def setup_data(self) -> tuple:
        """Generate and partition data."""
        print("Generating and partitioning data...")

        # Generate total data
        total_samples = self.num_clients * self.samples_per_client
        features, labels = generate_fraud_data(
            n_samples=total_samples,
            n_features=self.data_config['data']['num_features']
        )

        # Partition to clients
        client_data = partition_data_iid(
            features, labels, self.num_clients, self.samples_per_client
        )

        # Create test set
        _, _, test_loader = create_dataloaders(
            features, labels,
            batch_size=self.config['batch_size']
        )

        # Raw test features for ASR
        n_test = len(test_loader.dataset)
        test_features = features[-n_test:]
        test_labels = labels[-n_test:]

        return client_data, test_loader, test_features, test_labels

    def setup_clients(
        self,
        client_data: List[tuple],
        is_malicious: List[bool]
    ) -> List:
        """Create FL clients."""
        print(f"Creating {len(client_data)} clients...")

        clients = []

        for client_id, (features, labels) in enumerate(client_data):
            # Create model for client
            model = FraudMLP(input_dim=30)

            if is_malicious[client_id]:
                client = MaliciousClient(
                    client_id, model, features, labels,
                    self.config, self.attack_config['attack'],
                    device=self.device
                )
                print(f"  Client {client_id}: MALICIOUS")
            else:
                client = HonestClient(
                    client_id, model, features, labels,
                    self.config, device=self.device
                )

            clients.append(client)

        return clients

    def run_experiment(self):
        """Run complete backdoor attack experiment."""
        print("=" * 60)
        print("BACKDOOR ATTACK ON FEDERATED LEARNING")
        print("=" * 60)

        # Setup
        client_data, test_loader, test_features, test_labels = self.setup_data()

        # Create malicious client flags
        is_malicious = [i in self.malicious_ids for i in range(self.num_clients)]

        # Create clients
        clients = self.setup_clients(client_data, is_malicious)

        # Create server
        global_model = FraudMLP(input_dim=30)
        server = FlowerFLServer(
            global_model,
            self.num_clients,
            self.fed_config.get('client_fraction', 0.5),
            device=self.device
        )

        print(f"\nConfiguration:")
        print(f"  Total rounds: {self.num_rounds}")
        print(f"  Attack duration: {self.attack_duration}")
        print(f"  Malicious clients: {self.num_malicious}")
        print(f"  Trigger type: {self.attack_config['attack']['trigger_type']}")

        # Training with attack
        print("\n" + "=" * 60)
        print("PHASE 1: TRAINING WITH ATTACK")
        print("=" * 60)

        for round_idx in range(self.attack_duration):
            # Define client training function
            def client_train_fn(client_id, global_weights):
                return clients[client_id].train(global_weights)

            # Train one round
            metrics = server.fit_round(round_idx, client_train_fn)

            # Evaluate every 5 rounds
            if round_idx % 5 == 0 or round_idx == self.attack_duration - 1:
                eval_metrics = evaluate_backdoor_attack(
                    global_model, test_loader, test_features, test_labels,
                    self.attack_config['attack']
                )

                self.history['round'].append(round_idx)
                self.history['clean_accuracy'].append(eval_metrics['clean_accuracy'])
                self.history['attack_success_rate'].append(eval_metrics['attack_success_rate'])
                self.history['class_0_accuracy'].append(eval_metrics['class_0_accuracy'])
                self.history['class_1_accuracy'].append(eval_metrics['class_1_accuracy'])

                print(f"\nRound {round_idx}:")
                print(f"  Clean accuracy: {eval_metrics['clean_accuracy']:.4f}")
                print(f"  ASR: {eval_metrics['attack_success_rate']:.4f}")
                print(f"  Class 0 acc: {eval_metrics['class_0_accuracy']:.4f}")
                print(f"  Class 1 acc: {eval_metrics['class_1_accuracy']:.4f}")

        # Persistence testing
        print("\n" + "=" * 60)
        print("PHASE 2: PERSISTENCE TESTING")
        print("=" * 60)

        # Remove malicious clients
        honest_clients = [c for i, c in enumerate(clients) if not is_malicious[i]]
        print(f"Removed {self.num_malicious} malicious client(s)")
        print(f"Training with {len(honest_clients)} honest clients...")

        # Test persistence at different rounds
        persistence_rounds = self.attack_config['attack'].get('persistence_rounds', [5, 10, 20])

        persistence_results = test_backdoor_persistence(
            global_model, test_loader, test_features, test_labels,
            self.attack_config['attack'], persistence_rounds,
            server, honest_clients, self.device
        )

        # Save results
        self.save_results(persistence_results)

        # Plot results
        self.plot_results()

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {self.results_dir}")

        return self.history, persistence_results

    def save_results(self, persistence_results: Dict):
        """Save experiment results to file."""
        results = {
            'training_history': self.history,
            'persistence': persistence_results,
            'config': {
                'attack': self.attack_config['attack'],
                'data': self.data_config['data']
            }
        }

        import json
        results_path = self.results_dir / 'backdoor_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_path}")

    def plot_results(self):
        """Plot experiment results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Training history
        ax = axes[0]
        rounds = self.history['round']
        ax.plot(rounds, self.history['clean_accuracy'], 'b-o', label='Clean Accuracy', linewidth=2)
        ax.plot(rounds, self.history['attack_success_rate'], 'r-s', label='Attack Success Rate', linewidth=2)
        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Accuracy / ASR', fontsize=12)
        ax.set_title('Backdoor Attack: Clean Accuracy vs ASR', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Plot 2: Persistence
        ax = axes[1]
        if hasattr(self, 'persistence_results'):
            rounds = list(self.persistence_results.keys())
            asr_values = [self.persistence_results[r]['attack_success_rate'] for r in rounds]
            clean_values = [self.persistence_results[r]['clean_accuracy'] for r in rounds]

            ax.plot(rounds, asr_values, 'r-o', label='ASR after attack stops', linewidth=2)
            ax.plot(rounds, clean_values, 'b-s', label='Clean accuracy', linewidth=2)
            ax.set_xlabel('Rounds after attack stops', fontsize=12)
            ax.set_ylabel('Accuracy / ASR', fontsize=12)
            ax.set_title('Backdoor Persistence', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        plt.tight_layout()
        plot_path = self.results_dir / 'backdoor_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    experiment = BackdoorExperiment(device=device)
    history, persistence = experiment.run_experiment()
