"""
Baseline detection experiment.
Test anomaly detectors against known FL attacks (Days 15-17).
"""

import sys
import os
from typing import List, Tuple, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.detectors import (
    MagnitudeDetector,
    SimilarityDetector,
    LayerwiseDetector,
    HistoricalDetector,
    ClusteringDetector,
    SpectralDetector
)
from src.ensemble.voting_ensemble import VotingEnsemble
from src.evaluation.metrics import (
    compute_detection_metrics,
    plot_roc_curve,
    compare_detectors
)


def generate_synthetic_attacks(
    honest_updates: List[np.ndarray],
    attack_type: str = "backdoor",
    num_malicious: int = 3
) -> Tuple[List[np.ndarray], List[bool]]:
    """
    Generate synthetic malicious updates for testing.

    Args:
        honest_updates: Baseline honest updates
        attack_type: Type of attack to simulate
        num_malicious: Number of malicious clients

    Returns:
        Tuple of (all_updates, labels)
    """
    np.random.seed(42)
    malicious_updates = []

    for i in range(num_malicious):
        if attack_type == "backdoor":
            # Backdoor: Large update in specific direction
            honest_mean = np.mean(honest_updates, axis=0)
            attack_direction = np.random.randn(*honest_mean.shape)
            attack = honest_mean + attack_direction * 5.0
            malicious_updates.append(attack)

        elif attack_type == "label_flipping":
            # Label flipping: Opposite direction
            honest_mean = np.mean(honest_updates, axis=0)
            attack = -honest_mean + np.random.randn(*honest_mean.shape) * 0.5
            malicious_updates.append(attack)

        elif attack_type == "scaling":
            # Scaling attack: Multiply by large factor
            base_update = honest_updates[i % len(honest_updates)]
            attack = base_update * 5.0
            malicious_updates.append(attack)

    # Combine honest and malicious
    all_updates = honest_updates + malicious_updates
    labels = [False] * len(honest_updates) + [True] * num_malicious

    return all_updates, labels


def run_baseline_experiment(
    num_honest: int = 20,
    num_malicious: int = 5,
    attack_type: str = "backdoor"
) -> dict:
    """
    Run baseline detection experiment.

    Args:
        num_honest: Number of honest clients
        num_malicious: Number of malicious clients
        attack_type: Type of attack

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Baseline Detection Experiment: {attack_type}")
    print(f"{'='*60}\n")

    # Generate synthetic data
    print(f"Generating {num_honest} honest and {num_malicious} malicious updates...")
    np.random.seed(42)
    honest_updates = [np.random.randn(100) * 0.1 for _ in range(num_honest)]
    all_updates, labels = generate_synthetic_attacks(
        honest_updates, attack_type, num_malicious
    )

    # Create detectors
    print("Initializing detectors...")
    detectors = {
        'Magnitude': MagnitudeDetector(method="zscore", threshold=2.5),
        'Similarity': SimilarityDetector(similarity_threshold=0.7),
        'Clustering': ClusteringDetector(method="isolation_forest", contamination=0.1),
        'Spectral': SpectralDetector(n_components=5, threshold=2.5)
    }

    # Train/test split (use honest for training)
    train_honest = honest_updates[:int(0.7 * num_honest)]

    # Fit detectors on honest updates only
    print("Fitting detectors on honest updates...")
    for name, detector in detectors.items():
        detector.fit(train_honest)
        print(f"  - {name} fitted")

    # Compute scores for all clients
    print("\nComputing anomaly scores...")
    detector_scores = {}
    for name, detector in detectors.items():
        scores = [detector.compute_anomaly_score(update) for update in all_updates]
        detector_scores[name] = scores
        print(f"  - {name}: min={min(scores):.3f}, max={max(scores):.3f}")

    # Compare detectors
    print("\n" + "="*60)
    print("Detection Results (per detector):")
    print("="*60)
    results = compare_detectors(detector_scores, labels)

    for detector_name, metrics in results.items():
        print(f"\n{detector_name}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")
        print(f"  FPR:       {metrics['fpr']:.3f}")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")

    # Create ensemble
    print("\n" + "="*60)
    print("Ensemble Results:")
    print("="*60)

    fitted_detectors = list(detectors.values())
    ensemble = VotingEnsemble(detectors=fitted_detectors, voting="majority")

    ensemble_scores = [
        ensemble.compute_anomaly_score(update)
        for update in all_updates
    ]

    ensemble_preds = [
        ensemble.is_malicious(update)
        for update in all_updates
    ]

    ensemble_metrics = compute_detection_metrics(ensemble_preds, labels)
    print(f"\nVoting Ensemble (majority):")
    print(f"  Precision: {ensemble_metrics['precision']:.3f}")
    print(f"  Recall:    {ensemble_metrics['recall']:.3f}")
    print(f"  F1:        {ensemble_metrics['f1']:.3f}")
    print(f"  FPR:       {ensemble_metrics['fpr']:.3f}")
    print(f"  Accuracy:  {ensemble_metrics['accuracy']:.3f}")

    # Plot ROC curves
    print("\nGenerating ROC curves...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'ROC Curves - {attack_type} Attack', fontsize=14)

    for idx, (name, scores) in enumerate(detector_scores.items()):
        ax = axes[idx // 2, idx % 2]
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name} Detector')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"data/results/roc_{attack_type}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.close()

    return {
        'detector_scores': detector_scores,
        'ensemble_metrics': ensemble_metrics,
        'individual_metrics': results
    }


if __name__ == "__main__":
    # Run experiments for different attack types
    attack_types = ["backdoor", "label_flipping", "scaling"]

    all_results = {}
    for attack in attack_types:
        results = run_baseline_experiment(
            num_honest=20,
            num_malicious=5,
            attack_type=attack
        )
        all_results[attack] = results

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nROC curves saved to data/results/")
    print("Review results to see which detectors work best for each attack type.")
