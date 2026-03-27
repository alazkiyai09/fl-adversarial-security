"""
Ablation study experiment.
Measure contribution of each detector to ensemble performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.detectors import (
    MagnitudeDetector,
    SimilarityDetector,
    ClusteringDetector,
    SpectralDetector
)
from src.ensemble.voting_ensemble import VotingEnsemble
from src.evaluation.metrics import compute_detection_metrics


def run_ablation_study():
    """
    Test ensemble with each detector removed (ablation study).
    """
    print("\n" + "="*60)
    print("Ablation Study: Detector Contribution Analysis")
    print("="*60 + "\n")

    # Generate synthetic data
    np.random.seed(42)
    honest_updates = [np.random.randn(100) * 0.1 for _ in range(20)]

    # Add malicious updates (backdoor-style)
    malicious_updates = []
    for _ in range(5):
        honest_mean = np.mean(honest_updates, axis=0)
        attack_direction = np.random.randn(*honest_mean.shape)
        attack = honest_mean + attack_direction * 5.0
        malicious_updates.append(attack)

    all_updates = honest_updates + malicious_updates
    labels = [False] * 20 + [True] * 5

    # Train all detectors
    print("Training detectors...")
    detectors = {
        'Magnitude': MagnitudeDetector(method="zscore", threshold=2.5),
        'Similarity': SimilarityDetector(similarity_threshold=0.7),
        'Clustering': ClusteringDetector(method="isolation_forest", contamination=0.1),
        'Spectral': SpectralDetector(n_components=5, threshold=2.5)
    }

    for name, detector in detectors.items():
        detector.fit(honest_updates[:15])
        print(f"  - {name} fitted")

    # Test: Full ensemble
    print("\n" + "="*60)
    print("Testing Ensembles")
    print("="*60)

    results = {}

    # Full ensemble
    full_ensemble = VotingEnsemble(
        detectors=list(detectors.values()),
        voting="majority"
    )

    preds_full = [full_ensemble.is_malicious(u) for u in all_updates]
    metrics_full = compute_detection_metrics(preds_full, labels)

    results['Full Ensemble'] = metrics_full
    print(f"\nFull Ensemble: F1={metrics_full['f1']:.3f}, Acc={metrics_full['accuracy']:.3f}")

    # Ablation: Remove each detector one at a time
    for removed_name in detectors.keys():
        remaining_detectors = [
            det for name, det in detectors.items()
            if name != removed_name
        ]

        if not remaining_detectors:
            continue

        ensemble = VotingEnsemble(detectors=remaining_detectors, voting="majority")
        preds = [ensemble.is_malicious(u) for u in all_updates]
        metrics = compute_detection_metrics(preds, labels)

        ensemble_name = f"w/o {removed_name}"
        results[ensemble_name] = metrics
        print(f"{ensemble_name:20s}: F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}")

    # Compute contribution of each detector
    print("\n" + "="*60)
    print("Detector Contribution Analysis")
    print("="*60)

    contributions = {}
    for name in detectors.keys():
        ensemble_without = f"w/o {name}"
        if ensemble_without in results:
            # Contribution = Full F1 - (F1 without this detector)
            contribution = results['Full Ensemble']['f1'] - results[ensemble_without]['f1']
            contributions[name] = contribution

            impact = "critical" if contribution > 0.05 else "moderate" if contribution > 0.01 else "minimal"
            print(f"{name:20s}: +{contribution:.3f} F1 ({impact})")

    # Visualization
    print("\nGenerating visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: F1 Score Comparison
    names = list(results.keys())
    f1_scores = [results[name]['f1'] for name in names]

    colors = ['green' if name == 'Full Ensemble' else 'steelblue' for name in names]
    ax1.bar(range(len(names)), f1_scores, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score: Full vs Ablated Ensembles')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Detector Contributions
    if contributions:
        contrib_names = list(contributions.keys())
        contrib_values = list(contributions.values())

        colors = ['red' if v > 0.05 else 'orange' if v > 0.01 else 'gray' for v in contrib_values]
        ax2.barh(range(len(contrib_names)), contrib_values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(contrib_names)))
        ax2.set_yticklabels(contrib_names)
        ax2.set_xlabel('F1 Score Contribution')
        ax2.set_title('Individual Detector Contribution')
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_path = "data/results/ablation_study.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    most_critical = max(contributions, key=contributions.get)
    least_critical = min(contributions, key=contributions.get)

    print(f"Most critical detector: {most_critical} (+{contributions[most_critical]:.3f} F1)")
    print(f"Least critical detector: {least_critical} (+{contributions[least_critical]:.3f} F1)")

    return results, contributions


if __name__ == "__main__":
    run_ablation_study()
