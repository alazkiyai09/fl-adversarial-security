"""
Main evaluation script for Byzantine-robust aggregator evaluation.

This script runs comprehensive evaluation comparing all aggregators
against all attack types with varying attacker fractions.
"""

import os
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from pathlib import Path

from src.defenses.robust_aggregation.legacy.aggregators import (
    CoordinateWiseMedian,
    TrimmedMean,
    Krum,
    MultiKrum,
    Bulyan
)
from src.defenses.robust_aggregation.legacy.evaluation.metrics import (
    compute_accuracy,
    compute_attack_success_rate,
    compute_convergence_speed,
    compute_defense_effectiveness
)
from src.defenses.robust_aggregation.legacy.evaluation.comparison import (
    generate_comparison_matrix,
    generate_heatmap,
    generate_summary_table,
    plot_aggregator_performance,
    rank_aggregators
)


class RobustnessEvaluator:
    """
    Comprehensive evaluator for Byzantine-robust aggregators.

    Runs experiments comparing aggregators across:
    - Multiple attack types (label flipping, backdoor, model poisoning)
    - Varying attacker fractions (10%, 20%, 30%, 40%)
    - Multiple random seeds for statistical significance
    """

    def __init__(
        self,
        save_dir: str = None,
        device: str = 'cpu'
    ):
        """
        Initialize evaluator.

        Args:
            save_dir: Directory to save results and plots
            device: Device to run experiments on
        """
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path('./data/results')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize aggregators
        self.aggregators = {
            'Median': CoordinateWiseMedian(),
            'TrimmedMean': TrimmedMean(beta=0.2),
            'Krum': Krum(),
            'MultiKrum': MultiKrum(m=5),
            'Bulyan': Bulyan(),
        }

    def create_synthetic_model(self) -> nn.Module:
        """Create a simple MLP for testing."""
        return nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(self.device)

    def create_synthetic_data(
        self,
        n_samples: int = 1000,
        n_features: int = 784,
        n_classes: int = 10
    ) -> DataLoader:
        """Create synthetic dataset for testing."""
        # Generate random data
        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, n_classes, (n_samples,))

        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=32, shuffle=False)

    def simulate_client_updates(
        self,
        n_clients: int,
        num_attackers: int,
        attack_type: str
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Simulate client updates with and without attacks.

        Args:
            n_clients: Total number of clients
            num_attackers: Number of malicious clients
            attack_type: Type of attack to simulate

        Returns:
            List of model update dicts
        """
        updates = []
        model = self.create_synthetic_model()

        # Honest updates: centered around true gradient
        honest_gradient = {name: param.data.clone() + torch.randn_like(param.data) * 0.1
                          for name, param in model.named_parameters()}

        for i in range(n_clients):
            if i < num_attackers:
                # Malicious update
                if attack_type == 'label_flipping':
                    # Flip gradient direction
                    update = {name: -param * 2.0
                             for name, param in honest_gradient.items()}
                elif attack_type == 'backdoor':
                    # Add large bias toward specific direction
                    update = {name: param + torch.ones_like(param) * 10.0
                             for name, param in honest_gradient.items()}
                elif attack_type == 'model_poisoning':
                    # Add noise to distort gradient
                    update = {name: param + torch.randn_like(param) * 5.0
                             for name, param in honest_gradient.items()}
                else:
                    update = honest_gradient.copy()
            else:
                # Honest update with small noise
                update = {name: param + torch.randn_like(param) * 0.05
                         for name, param in honest_gradient.items()}

            updates.append(update)

        return updates

    def evaluate_aggregator(
        self,
        aggregator_name: str,
        attack_type: str,
        attacker_fraction: float,
        n_clients: int = 20,
        n_rounds: int = 10,
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Evaluate a single aggregator against a single attack configuration.

        Args:
            aggregator_name: Name of aggregator to test
            attack_type: Type of attack
            attacker_fraction: Fraction of malicious clients (0.0 to 1.0)
            n_clients: Total number of clients
            n_rounds: Number of training rounds
            seed: Random seed for reproducibility

        Returns:
            Dict with evaluation metrics
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        aggregator = self.aggregators[aggregator_name]
        num_attackers = int(n_clients * attacker_fraction)

        # Simulate federated learning
        model = self.create_synthetic_model()
        test_loader = self.create_synthetic_data()

        accuracy_history = []

        for round_idx in range(n_rounds):
            # Get client updates
            updates = self.simulate_client_updates(
                n_clients, num_attackers, attack_type
            )

            # Aggregate
            try:
                aggregated = aggregator.aggregate(updates, num_attackers)

                # Apply aggregated update (simplified: just add to model)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in aggregated:
                            param.data += aggregated[name] * 0.01

                # Evaluate
                acc = compute_accuracy(model, test_loader, self.device)
                accuracy_history.append(acc)

            except ValueError as e:
                # Aggregator failed (e.g., insufficient clients)
                print(f"  {aggregator_name} failed: {e}")
                return {
                    'accuracy': 0.0,
                    'asr': 1.0,
                    'convergence_speed': n_rounds,
                    'final_accuracy': 0.0,
                }

        # Compute metrics
        final_accuracy = accuracy_history[-1] if accuracy_history else 0.0
        asr = compute_attack_success_rate(
            model, test_loader, attack_type, self.device
        )
        convergence = compute_convergence_speed(accuracy_history, target_threshold=0.5)

        return {
            'accuracy': final_accuracy,
            'asr': asr,
            'convergence_speed': convergence,
            'final_accuracy': final_accuracy,
        }

    def run_full_evaluation(
        self,
        attacker_fractions: List[float] = [0.1, 0.2, 0.3, 0.4],
        attack_types: List[str] = ['label_flipping', 'backdoor', 'model_poisoning'],
        aggregator_names: List[str] = None,
        n_clients: int = 20,
        n_rounds: int = 10,
        n_seeds: int = 3
    ) -> Dict:
        """
        Run comprehensive evaluation of all aggregators.

        Args:
            attacker_fractions: List of attacker fractions to test
            attack_types: List of attack types to test
            aggregator_names: List of aggregators to test (None = all)
            n_clients: Total number of clients per round
            n_rounds: Number of training rounds
            n_seeds: Number of random seeds for statistical significance

        Returns:
            Nested dict with results
        """
        if aggregator_names is None:
            aggregator_names = list(self.aggregators.keys())

        results = {
            agg: {
                attack: {
                    str(fraction): [] for fraction in attacker_fractions
                }
                for attack in attack_types
            }
            for agg in aggregator_names
        }

        print("Starting robust aggregator evaluation...")
        print(f"Aggregators: {aggregator_names}")
        print(f"Attacks: {attack_types}")
        print(f"Attacker fractions: {attacker_fractions}")
        print(f"Seeds per config: {n_seeds}")
        print("-" * 60)

        # Run experiments
        for agg_name in aggregator_names:
            for attack_type in attack_types:
                for fraction in attacker_fractions:
                    print(f"Testing {agg_name} vs {attack_type} @ {fraction:.0%}")

                    seed_results = []
                    for seed_idx in range(n_seeds):
                        seed = 42 + seed_idx
                        result = self.evaluate_aggregator(
                            agg_name, attack_type, fraction,
                            n_clients, n_rounds, seed
                        )
                        seed_results.append(result)

                    # Average across seeds
                    avg_result = {
                        metric: np.mean([r[metric] for r in seed_results])
                        for metric in seed_results[0].keys()
                    }

                    results[agg_name][attack_type][str(fraction)] = avg_result

                    print(f"  Accuracy: {avg_result['accuracy']:.3f}, "
                          f"ASR: {avg_result['asr']:.3f}")

        # Save results
        self._save_results(results)

        # Generate visualizations
        self._generate_plots(results, attacker_fractions, attack_types)

        print("\nEvaluation complete!")
        print(f"Results saved to: {self.save_dir}")

        return results

    def _save_results(self, results: Dict) -> None:
        """Save results to JSON file."""
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        results_path = self.save_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, default=convert, indent=2)

        print(f"Results saved to: {results_path}")

    def _generate_plots(
        self,
        results: Dict,
        attacker_fractions: List[float],
        attack_types: List[str]
    ) -> None:
        """Generate and save visualization plots."""
        # Accuracy heatmap
        acc_matrix = generate_comparison_matrix(results, metric='accuracy')
        fig = generate_heatmap(
            acc_matrix,
            metric='accuracy',
            save_path=self.save_dir / 'accuracy_heatmap.png'
        )
        if fig:
            plt.close(fig)

        # ASR heatmap
        asr_matrix = generate_comparison_matrix(results, metric='asr')
        fig = generate_heatmap(
            asr_matrix,
            metric='asr',
            save_path=self.save_dir / 'asr_heatmap.png'
        )
        if fig:
            plt.close(fig)

        # Performance line plots
        fig = plot_aggregator_performance(
            results,
            aggregator_names=list(results.keys()),
            metric='accuracy',
            save_path=self.save_dir / 'performance_curves.png'
        )
        if fig:
            plt.close(fig)

        # Ranking table
        ranking = rank_aggregators(results, metric='accuracy', higher_is_better=True)
        ranking_path = self.save_dir / 'aggregator_ranking.csv'
        ranking.to_csv(ranking_path)
        print(f"Ranking saved to: {ranking_path}")


def main():
    """Main entry point for evaluation."""
    evaluator = RobustnessEvaluator(
        save_dir='./data/results',
        device='cpu'
    )

    results = evaluator.run_full_evaluation(
        attacker_fractions=[0.1, 0.2, 0.3, 0.4],
        attack_types=['label_flipping', 'backdoor', 'model_poisoning'],
        aggregator_names=None,  # Test all aggregators
        n_clients=20,
        n_rounds=10,
        n_seeds=3
    )

    return results


if __name__ == '__main__':
    results = main()
