"""
Experiment: Test Defenses Against Membership Inference Attacks

This script evaluates the effectiveness of various defenses:
1. Differential Privacy
2. Training epochs (overfitting vs vulnerability)
3. Regularization strength

Usage:
    python experiments/experiment_defenses.py --config config/attack_config.yaml
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import TensorDataset

sys.path.append('src')

from utils.data_splits import DataSplitter
from target_models.fl_target import FraudDetectionNN, create_client_splits
from defenses.dp_defense import DPTargetTrainer, test_dp_defense, analyze_privacy_utility_tradeoff
from attacks.metric_attacks import loss_based_attack
from evaluation.attack_metrics import compute_attack_metrics, print_attack_results


def create_synthetic_dataset(n_samples=5000, n_features=20, random_seed=42):
    """Create synthetic fraud detection dataset."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    X = torch.randn(n_samples, n_features)
    fraud_prob = 0.1
    y = torch.zeros(n_samples, dtype=torch.long)
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * fraud_prob), replace=False)
    y[fraud_indices] = 1

    return TensorDataset(X, y)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Test defenses against membership inference attacks')
    parser.add_argument('--config', type=str, default='config/attack_config.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--n_features', type=int, default=20)
    parser.add_argument('--save_results', action='store_true')

    args = parser.parse_args()

    config = load_config(args.config)
    random_seed = config['evaluation']['random_seed']

    print("="*80)
    print("DEFENSE EVALUATION: MEMBERSHIP INFERENCE ATTACKS")
    print("="*80)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # ========================================================================
    # Setup: Load and split data
    # ========================================================================
    print("\nLoading and splitting data...")

    full_dataset = create_synthetic_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        random_seed=random_seed
    )

    splitter = DataSplitter(
        full_dataset=full_dataset,
        config_path=args.config,
        random_seed=random_seed
    )

    splits = splitter.create_splits()
    member_loader, nonmember_loader = splitter.create_attack_test_split(n_samples=500)

    # Create FL client datasets
    client_datasets = create_client_splits(
        full_dataset=splits['target_train'],
        n_clients=10,
        batch_size=32,
        random_seed=random_seed
    )

    # ========================================================================
    # Experiment 1: Differential Privacy Defense
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: DIFFERENTIAL PRIVACY DEFENSE")
    print("="*80)

    noise_levels = config['defenses']['differential_privacy']['noise_levels']

    dp_results = test_dp_defense(
        base_model_class=FraudDetectionNN,
        model_config={
            'input_dim': args.n_features,
            'hidden_dims': [64, 32],
            'num_classes': 2
        },
        client_datasets=client_datasets,
        member_data=member_loader,
        nonmember_data=nonmember_loader,
        attack_fn=loss_based_attack,
        noise_levels=noise_levels,
        n_rounds=10,
        n_clients=10,
        device=args.device
    )

    print("\n" + "="*80)
    print("DP DEFENSE SUMMARY")
    print("="*80)

    print(f"\n{'Noise':<12} {'AUC':<10} {'Accuracy':<10} {'Improvement':<15}")
    print("-" * 50)

    for noise_level, metrics in sorted(dp_results.items()):
        auc = metrics['auc']
        acc = metrics['accuracy']
        vs_random = ((auc - 0.5) / 0.5) * 100

        print(f"{noise_level:<12.2f} {auc:<10.4f} {acc:<10.4f} {vs_random:>+14.1f}%")

    # Plot privacy-utility tradeoff
    if args.save_results:
        analyze_privacy_utility_tradeoff(
            dp_results=dp_results,
            save_path='results/attack_performance/dp_defense_tradeoff.png'
        )

    # ========================================================================
    # Experiment 2: Training Epochs vs Vulnerability
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: TRAINING EPOCHS vs VULNERABILITY")
    print("="*80)

    epochs_to_test = config['defenses']['training_epochs']['epochs_to_test']
    epoch_results = {}

    for n_rounds in epochs_to_test:
        print(f"\n{'='*80}")
        print(f"Testing with {n_rounds} FL rounds")
        print(f"{'='*80}")

        model = FraudDetectionNN(
            input_dim=args.n_features,
            hidden_dims=[64, 32],
            num_classes=2
        )

        from target_models.fl_target import FLTargetTrainer
        trainer = FLTargetTrainer(
            model=model,
            n_clients=10,
            local_epochs=5,
            client_lr=0.01,
            device=args.device
        )

        trained_model = trainer.train_fl_model(
            client_datasets=client_datasets,
            n_rounds=n_rounds,
            verbose=False
        )

        all_scores, true_labels, _ = loss_based_attack(
            target_model=trained_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device=args.device
        )

        metrics = compute_attack_metrics(all_scores, true_labels)
        epoch_results[n_rounds] = metrics

        print(f"\nResults (rounds={n_rounds}):")
        print(f"  AUC: {metrics['auc']:.4f}")

    print("\n" + "="*80)
    print("EPOCHS vs VULNERABILITY SUMMARY")
    print("="*80)

    print(f"\n{'Rounds':<12} {'AUC':<10} {'Accuracy':<10} {'vs Random':<15}")
    print("-" * 50)

    for n_rounds, metrics in sorted(epoch_results.items()):
        auc = metrics['auc']
        acc = metrics['accuracy']
        vs_random = ((auc - 0.5) / 0.5) * 100

        print(f"{n_rounds:<12} {auc:<10.4f} {acc:<10.4f} {vs_random:>+14.1f}%")

    # Plot epochs vs vulnerability
    if args.save_results:
        import matplotlib.pyplot as plt

        rounds = sorted(epoch_results.keys())
        aucs = [epoch_results[r]['auc'] for r in rounds]

        plt.figure(figsize=(8, 6))
        plt.plot(rounds, aucs, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Random Baseline')
        plt.xlabel('FL Training Rounds', fontsize=12)
        plt.ylabel('Attack AUC', fontsize=12)
        plt.title('Vulnerability vs Training Duration', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig('results/attack_performance/vulnerability_vs_rounds.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to results/attack_performance/vulnerability_vs_rounds.png")
        plt.close()

    # ========================================================================
    # Final Summary and Recommendations
    # ========================================================================
    print("\n" + "="*80)
    print("DEFENSE EVALUATION SUMMARY")
    print("="*80)

    print("\n1. Differential Privacy:")
    best_noise = max(dp_results.items(), key=lambda x: x[1]['auc'])
    worst_noise = min(dp_results.items(), key=lambda x: x[1]['auc'])

    print(f"   Best defense (noise={worst_noise[0]}): AUC = {worst_noise[1]['auc']:.4f}")
    print(f"   Worst defense (noise={best_noise[0]}): AUC = {best_noise[1]['auc']:.4f}")

    if worst_noise[1]['auc'] < 0.6:
        print("   ✓ DP is EFFECTIVE at mitigating attacks")
    else:
        print("   ✗ DP may need stronger parameters")

    print("\n2. Training Duration:")
    early_rounds = min(epoch_results.items(), key=lambda x: x[0])
    late_rounds = max(epoch_results.items(), key=lambda x: x[0])

    print(f"   Early stopping ({early_rounds[0]} rounds): AUC = {early_rounds[1]['auc']:.4f}")
    print(f"   Extended training ({late_rounds[0]} rounds): AUC = {late_rounds[1]['auc']:.4f}")

    if late_rounds[1]['auc'] > early_rounds[1]['auc']:
        print("   ⚠ Longer training INCREASES vulnerability")
    else:
        print("   ✓ Longer training does NOT significantly increase vulnerability")

    print("\n3. Recommendations:")

    if worst_noise[1]['auc'] < 0.6:
        print("   • Use Differential Privacy with noise >= {:.1f}".format(worst_noise[0]))
    else:
        print("   • Consider stronger DP (higher noise, stricter clipping)")

    if late_rounds[1]['auc'] - early_rounds[1]['auc'] > 0.1:
        print("   • Implement early stopping to reduce vulnerability")
        print(f"   • Stop at ~{early_rounds[0]} rounds instead of {late_rounds[0]}")

    print("   • Monitor attack metrics during model development")
    print("   • Consider adversarial training with membership privacy objectives")

    print("\n" + "="*80)
    print("Defense evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
