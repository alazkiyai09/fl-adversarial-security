"""
Experiment: Shadow Model-Based Membership Inference Attack

This script executes the complete shadow model attack pipeline:
1. Load/split data
2. Train shadow models
3. Generate attack training data
4. Train attack model
5. Evaluate attack on target model
6. Generate visualizations and report

Usage:
    python experiments/run_shadow_attack.py --config config/attack_config.yaml
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
from target_models.fl_target import FraudDetectionNN
from attacks.shadow_models import (
    ShadowModelTrainer,
    train_attack_model,
    shadow_model_attack
)
from evaluation.attack_metrics import (
    compute_attack_metrics,
    print_attack_results,
    plot_roc_curve,
    plot_pr_curve,
    plot_score_distributions,
    vulnerability_analysis
)


def create_synthetic_dataset(n_samples=5000, n_features=20, random_seed=42):
    """
    Create synthetic fraud detection dataset for experimentation.

    In practice, replace this with actual data loading.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_seed: Random seed

    Returns:
        TensorDataset
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Generate synthetic features
    X = torch.randn(n_samples, n_features)

    # Generate synthetic labels (imbalanced: 10% fraud)
    fraud_prob = 0.1
    y = torch.zeros(n_samples, dtype=torch.long)
    fraud_indices = np.random.choice(
        n_samples,
        size=int(n_samples * fraud_prob),
        replace=False
    )
    y[fraud_indices] = 1

    print(f"✓ Created synthetic dataset:")
    print(f"  - Total samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - Fraud rate: {fraud_prob * 100:.1f}%")

    return TensorDataset(X, y)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Run shadow model-based membership inference attack')
    parser.add_argument('--config', type=str, default='config/attack_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of samples in dataset')
    parser.add_argument('--n_features', type=int, default=20,
                        help='Number of features')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to disk')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    random_seed = config['evaluation']['random_seed']

    print("="*80)
    print("SHADOW MODEL-BASED MEMBERSHIP INFERENCE ATTACK")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Features: {args.n_features}")
    print(f"  Random seed: {random_seed}")

    # Set random seeds
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
    split_sizes = splitter.get_split_sizes()

    print(f"\nData split sizes:")
    for name, size in split_sizes.items():
        print(f"  {name}: {size}")

    # ========================================================================
    # STEP 2: Train shadow models
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Train Shadow Models")
    print("="*80)

    n_shadow = config['shadow_models']['n_shadow']
    shadow_epochs = config['shadow_models']['shadow_epochs']
    shadow_lr = config['shadow_models']['shadow_lr']

    # Create shadow model splits
    shadow_splits = splitter.create_shadow_model_splits(n_shadow=n_shadow)

    # Initialize shadow model trainer
    shadow_trainer = ShadowModelTrainer(
        model_architecture=FraudDetectionNN,
        n_shadow=n_shadow,
        shadow_epochs=shadow_epochs,
        learning_rate=shadow_lr,
        device=args.device,
        random_seed=random_seed
    )

    # Train shadow models
    shadow_models = shadow_trainer.train_all_shadow_models(
        shadow_splits=shadow_splits,
        model_config={
            'input_dim': args.n_features,
            'hidden_dims': [64, 32],
            'num_classes': 2
        },
        verbose=True
    )

    # ========================================================================
    # STEP 3: Generate attack training data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Generate Attack Training Data")
    print("="*80)

    from attacks.shadow_models import generate_attack_training_data

    attack_features, attack_labels = generate_attack_training_data(
        shadow_models=shadow_models,
        shadow_splits=shadow_splits,
        device=args.device
    )

    # ========================================================================
    # STEP 4: Train attack model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Train Attack Model")
    print("="*80)

    attack_model_type = config.get('attack_model_type', 'random_forest')

    attack_model = train_attack_model(
        attack_features=attack_features,
        attack_labels=attack_labels,
        attack_model_type=attack_model_type
    )

    # ========================================================================
    # STEP 5: Create and attack target model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Attack Target Model")
    print("="*80)

    # Create target model (trained on target_train data)
    print("\nTraining target model...")

    from target_models.fl_target import FLTargetTrainer, create_client_splits

    target_model = FraudDetectionNN(
        input_dim=args.n_features,
        hidden_dims=[64, 32],
        num_classes=2
    )

    # Create FL client splits from target_train data
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

    # Train FL model (fewer rounds for faster experimentation)
    trained_target_model = fl_trainer.train_fl_model(
        client_datasets=client_datasets,
        n_rounds=10,
        verbose=True
    )

    # Create attack test data
    member_loader, nonmember_loader = splitter.create_attack_test_split(n_samples=500)

    # Execute attack
    print("\nExecuting shadow model attack...")

    all_scores, true_labels, (member_scores, nonmember_scores) = shadow_model_attack(
        target_model=trained_target_model,
        attack_model=attack_model,
        member_data=member_loader,
        nonmember_data=nonmember_loader,
        device=args.device
    )

    # ========================================================================
    # STEP 6: Evaluate attack performance
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Evaluate Attack Performance")
    print("="*80)

    fpr_points = config['evaluation']['fpr_points']
    metrics = compute_attack_metrics(all_scores, true_labels, fpr_points=fpr_points)

    print_attack_results(metrics, "Shadow Model Attack")

    # ========================================================================
    # STEP 7: Visualizations and analysis
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: Generate Visualizations")
    print("="*80)

    results_dir = 'results/attack_performance'

    if args.save_results:
        # ROC curve
        fpr, tpr, _ = torch.zeros(0), torch.zeros(0), torch.zeros(0)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(true_labels, all_scores)

        plot_roc_curve(
            fpr=fpr,
            tpr=tpr,
            roc_auc=metrics['auc'],
            save_path=os.path.join(results_dir, 'shadow_attack_roc.png'),
            attack_name="Shadow Model Attack"
        )

        # PR curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(true_labels, all_scores)

        plot_pr_curve(
            precision=precision,
            recall=recall,
            pr_auc=metrics['pr_auc'],
            save_path=os.path.join(results_dir, 'shadow_attack_pr.png'),
            attack_name="Shadow Model Attack"
        )

        # Score distributions
        plot_score_distributions(
            member_scores=member_scores,
            nonmember_scores=nonmember_scores,
            save_path=os.path.join(results_dir, 'shadow_attack_distributions.png'),
            attack_name="Shadow Model Attack"
        )

        # Vulnerability analysis
        print("\nVulnerability Analysis:")
        vulnerability_analysis(
            membership_scores=all_scores,
            true_labels=true_labels
        )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("ATTACK SUMMARY")
    print("="*80)

    print(f"\nTarget Model:")
    print(f"  Architecture: FraudDetectionNN")
    print(f"  Input features: {args.n_features}")
    print(f"  FL rounds: 10")

    print(f"\nShadow Models:")
    print(f"  Number of shadow models: {n_shadow}")
    print(f"  Training epochs: {shadow_epochs}")

    print(f"\nAttack Results:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")

    # Compare to random baseline
    if metrics['auc'] > 0.5:
        improvement = (metrics['auc'] - 0.5) / 0.5 * 100
        print(f"\n✓ Attack SUCCESSFUL: {improvement:.1f}% better than random guessing")
    else:
        print(f"\n✗ Attack FAILED: Performs worse than random guessing")

    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)


if __name__ == "__main__":
    main()
