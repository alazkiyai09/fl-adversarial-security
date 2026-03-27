"""
Experiment: Threshold-Based and Metric-Based Membership Inference Attacks

This script executes simple threshold and metric-based attacks:
1. Confidence-based attacks (max, mean, entropy)
2. Loss-based attacks
3. Modified entropy attacks

Usage:
    python experiments/run_threshold_attack.py --config config/attack_config.yaml
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import TensorDataset

# Add src to path
sys.path.append('src')

from utils.data_splits import DataSplitter
from target_models.fl_target import FraudDetectionNN, FLTargetTrainer, create_client_splits
from attacks.threshold_attack import confidence_based_attack, find_optimal_threshold
from attacks.metric_attacks import (
    loss_based_attack,
    entropy_based_attack,
    modified_entropy_attack,
    aggregate_metric_attacks
)
from evaluation.attack_metrics import (
    compute_attack_metrics,
    print_attack_results,
    plot_roc_curve,
    plot_pr_curve,
    plot_score_distributions,
    compare_attacks
)


def create_synthetic_dataset(n_samples=5000, n_features=20, random_seed=42):
    """Create synthetic fraud detection dataset."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    X = torch.randn(n_samples, n_features)
    fraud_prob = 0.1
    y = torch.zeros(n_samples, dtype=torch.long)
    fraud_indices = np.random.choice(
        n_samples,
        size=int(n_samples * fraud_prob),
        replace=False
    )
    y[fraud_indices] = 1

    return TensorDataset(X, y)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Run threshold-based membership inference attacks')
    parser.add_argument('--config', type=str, default='config/attack_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of samples')
    parser.add_argument('--n_features', type=int, default=20,
                        help='Number of features')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to disk')

    args = parser.parse_args()

    config = load_config(args.config)
    random_seed = config['evaluation']['random_seed']

    print("="*80)
    print("THRESHOLD-BASED AND METRIC-BASED MEMBERSHIP INFERENCE ATTACKS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Features: {args.n_features}")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # ========================================================================
    # STEP 1: Load and split data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Load and Split Data")
    print("="*80)

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

    # ========================================================================
    # STEP 2: Train target FL model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Train Target FL Model")
    print("="*80)

    target_model = FraudDetectionNN(
        input_dim=args.n_features,
        hidden_dims=[64, 32],
        num_classes=2
    )

    client_datasets = create_client_splits(
        full_dataset=splits['target_train'],
        n_clients=10,
        batch_size=32,
        random_seed=random_seed
    )

    fl_trainer = FLTargetTrainer(
        model=target_model,
        n_clients=10,
        local_epochs=5,
        client_lr=0.01,
        device=args.device
    )

    trained_target_model = fl_trainer.train_fl_model(
        client_datasets=client_datasets,
        n_rounds=10,
        verbose=True
    )

    # ========================================================================
    # STEP 3: Create attack test data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Create Attack Test Data")
    print("="*80)

    member_loader, nonmember_loader = splitter.create_attack_test_split(n_samples=500)

    print(f"Member samples: {len(member_loader.dataset)}")
    print(f"Non-member samples: {len(nonmember_loader.dataset)}")

    # ========================================================================
    # STEP 4: Run confidence-based attacks
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Confidence-Based Attacks")
    print("="*80)

    confidence_results = {}

    for conf_type in ['max', 'mean', 'entropy']:
        print(f"\nRunning {conf_type} confidence attack...")

        all_scores, true_labels, (member_scores, nonmember_scores) = confidence_based_attack(
            target_model=trained_target_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device=args.device,
            confidence_type=conf_type
        )

        metrics = compute_attack_metrics(true_labels, true_labels)
        confidence_results[f'confidence_{conf_type}'] = {
            'metrics': metrics,
            'scores': all_scores,
            'labels': true_labels,
            'member_scores': member_scores,
            'nonmember_scores': nonmember_scores
        }

        print_attack_results(metrics, f"{conf_type.upper()} Confidence Attack")

    # ========================================================================
    # STEP 5: Run metric-based attacks
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Metric-Based Attacks")
    print("="*80)

    metric_results = aggregate_metric_attacks(
        target_model=trained_target_model,
        member_data=member_loader,
        nonmember_data=nonmember_loader,
        device=args.device,
        attacks=['loss', 'entropy', 'modified_entropy']
    )

    for attack_name, (all_scores, true_labels, (member_vals, nonmember_vals)) in metric_results.items():
        metrics = compute_attack_metrics(all_scores, true_labels)

        metric_results[attack_name] = {
            'metrics': metrics,
            'scores': all_scores,
            'labels': true_labels,
            'member_scores': member_vals,
            'nonmember_scores': nonmember_vals
        }

        print_attack_results(metrics, f"{attack_name.replace('_', ' ').title()} Attack")

    # ========================================================================
    # STEP 6: Compare all attacks
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Compare All Attacks")
    print("="*80)

    # Combine all results
    all_results = {**confidence_results, **metric_results}

    # Print comparison table
    print(f"\n{'Attack':<30} {'AUC':<10} {'Accuracy':<10} {'PR-AUC':<10}")
    print("-" * 60)

    best_attack = None
    best_auc = 0.0

    for attack_name, results in all_results.items():
        metrics = results['metrics']
        print(f"{attack_name:<30} {metrics['auc']:<10.4f} {metrics['accuracy']:<10.4f} {metrics['pr_auc']:<10.4f}")

        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_attack = attack_name

    print(f"\nBest attack: {best_attack} (AUC = {best_auc:.4f})")

    # ========================================================================
    # STEP 7: Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: Generate Visualizations")
    print("="*80)

    results_dir = 'results/attack_performance'

    if args.save_results:
        # ROC curves for all attacks
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        plt.figure(figsize=(10, 8))

        for attack_name, results in all_results.items():
            fpr, tpr, _ = roc_curve(results['labels'], results['scores'])
            auc = results['metrics']['auc']
            plt.plot(fpr, tpr, linewidth=2, label=f"{attack_name} (AUC = {auc:.3f})")

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves: All Threshold-Based Attacks', fontsize=14)
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(results_dir, 'all_attacks_roc.png'), dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves saved to {os.path.join(results_dir, 'all_attacks_roc.png')}")
        plt.close()

        # Comparison bar chart
        compare_attacks(
            {name: r['metrics'] for name, r in all_results.items()},
            metric='auc',
            save_path=os.path.join(results_dir, 'attack_comparison_auc.png')
        )

        # Score distributions for best attack
        if best_attack:
            plot_score_distributions(
                member_scores=all_results[best_attack]['member_scores'],
                nonmember_scores=all_results[best_attack]['nonmember_scores'],
                save_path=os.path.join(results_dir, f'{best_attack}_distributions.png'),
                attack_name=best_attack.replace('_', ' ').title()
            )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("ATTACK SUMMARY")
    print("="*80)

    print(f"\nTotal attacks evaluated: {len(all_results)}")
    print(f"Best attack: {best_attack}")
    print(f"Best AUC: {best_auc:.4f}")

    if best_auc > 0.5:
        improvement = (best_auc - 0.5) / 0.5 * 100
        print(f"\n✓ At least one attack SUCCESSFUL: {improvement:.1f}% better than random")
    else:
        print(f"\n✗ All attacks FAILED: No attack better than random")

    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)


if __name__ == "__main__":
    main()
