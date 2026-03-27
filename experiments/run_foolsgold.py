#!/usr/bin/env python3
"""
Main entry point for FoolsGold defense experiments.

Usage:
    python run.py --mode experiment --defense foolsgold --attack sybil
    python run.py --mode ablation
    python run.py --mode test
"""

import argparse
import sys
from src.experiments import run_defense_comparison, run_single_experiment, run_ablation_study


def main():
    parser = argparse.ArgumentParser(description="FoolsGold Defense Experiments")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["experiment", "ablation", "test"],
        default="experiment",
        help="Execution mode"
    )

    parser.add_argument(
        "--defense",
        type=str,
        choices=["foolsgold", "krum", "multi_krum", "trimmed_mean", "fedavg"],
        default="foolsgold",
        help="Defense strategy"
    )

    parser.add_argument(
        "--attack",
        type=str,
        choices=["sybil", "collusion", "none"],
        default="sybil",
        help="Attack type"
    )

    parser.add_argument(
        "--num-malicious",
        type=int,
        default=2,
        help="Number of malicious clients"
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=50,
        help="Number of training rounds"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    print("="*80)
    print("FoolsGold Defense: Sybil-Resistant Federated Learning")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Defense: {args.defense}")
    print(f"Attack: {args.attack}")
    print(f"Malicious Clients: {args.num_malicious}")
    print(f"Rounds: {args.num_rounds}")
    print("="*80)

    if args.mode == "test":
        # Run tests
        import subprocess
        result = subprocess.run(["pytest", "tests/", "-v"])
        sys.exit(result.returncode)

    elif args.mode == "ablation":
        # Run ablation study
        results = run_ablation_study(output_dir=args.output_dir)
        print("\nAblation study completed!")
        print(f"Results saved to {args.output_dir}/ablation")

    elif args.mode == "experiment":
        # Run single experiment
        metrics = run_single_experiment(
            defense=args.defense,
            attack_type=args.attack,
            num_malicious=args.num_malicious,
            num_rounds=args.num_rounds
        )

        print("\n" + "="*80)
        print("EXPERIMENT RESULTS")
        print("="*80)
        print(f"Final Accuracy: {metrics.get('final_accuracy', 0):.4f}")
        print(f"Final Loss: {metrics.get('final_loss', 0):.4f}")
        print(f"Attack Success Rate: {metrics.get('final_attack_success', 0):.4f}")

        if 'detection_precision' in metrics:
            print(f"\nDetection Metrics:")
            print(f"  Precision: {metrics['detection_precision']:.4f}")
            print(f"  Recall: {metrics['detection_recall']:.4f}")
            print(f"  F1 Score: {metrics['detection_f1']:.4f}")

        print("="*80)


if __name__ == "__main__":
    main()
