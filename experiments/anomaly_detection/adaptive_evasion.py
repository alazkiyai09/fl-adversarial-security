"""
Adaptive evasion experiment.
Test detector robustness against adaptive attackers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.detectors import MagnitudeDetector, SimilarityDetector
from src.attacks.adaptive_attacker import AdaptiveAttacker
from src.evaluation.metrics import compute_detection_metrics


def run_adaptive_evasion_experiment():
    """
    Test detectors against adaptive attackers who know detection thresholds.
    """
    print("\n" + "="*60)
    print("Adaptive Evasion Experiment")
    print("="*60 + "\n")

    # Generate honest updates
    np.random.seed(42)
    honest_updates = [np.random.randn(100) * 0.1 for _ in range(20)]

    # Train detector
    detector = MagnitudeDetector(method="zscore", threshold=2.5)
    detector.fit(honest_updates[:15])

    # Test different adaptive strategies
    strategies = ["threshold_aware", "gradual", "camouflage"]
    results = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing against: {strategy} attack")
        print(f"{'='*60}")

        # Create adaptive attacker
        attacker = AdaptiveAttacker(
            strategy=strategy,
            detection_threshold=detector.threshold
        )

        # Simulate multiple rounds
        predictions_per_round = []
        for round_num in range(10):
            # Generate malicious update for this round
            honest_base = honest_updates[round_num % len(honest_updates)]
            malicious_update = attacker.generate_attack(
                honest_base,
                attack_magnitude=5.0
            )

            # Detect
            is_malicious = detector.is_malicious(malicious_update)
            predictions_per_round.append(is_malicious)

            detected = "DETECTED" if is_malicious else "evaded"
            print(f"  Round {round_num + 1}: {detected}")

        # Compute detection rate over rounds
        detection_rate = sum(predictions_per_round) / len(predictions_per_round)
        results[strategy] = {
            'detection_rate': detection_rate,
            'predictions_per_round': predictions_per_round
        }

        print(f"\nDetection rate: {detection_rate:.2%}")

    # Plot detection over rounds
    print("\n" + "="*60)
    print("Generating visualization...")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Adaptive Attack Evasion Over Rounds', fontsize=14)

    for idx, (strategy, result) in enumerate(results.items()):
        ax = axes[idx]

        rounds = list(range(1, 11))
        detected = [int(p) for p in result['predictions_per_round']]

        # Cumulative detection
        cumulative_detected = np.cumsum(detected)
        detection_percentage = [
            d / (r + 1) for r, d in enumerate(cumulative_detected)
        ]

        ax.plot(rounds, detection_percentage, 'b-o', linewidth=2, markersize=6)
        ax.axhline(y=0.5, color='r', linestyle='--', label='50% detection')
        ax.set_xlabel('Round')
        ax.set_ylabel('Cumulative Detection Rate')
        ax.set_title(f'{strategy.replace("_", " ").title()}')
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "data/results/adaptive_evasion.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for strategy, result in results.items():
        print(f"{strategy:20s}: {result['detection_rate']:.2%} detection rate")

    print("\nKey findings:")
    print("- threshold_aware: Harder to detect (stays below threshold)")
    print("- gradual: Starts undetected, may be caught eventually")
    print("- camouflage: Mimics honest, moderate detection difficulty")

    return results


if __name__ == "__main__":
    run_adaptive_evasion_experiment()
