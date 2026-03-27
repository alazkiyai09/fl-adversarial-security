"""
Scalability experiments for secure aggregation.

Analyzes how the protocol performs with varying numbers of clients.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.secure_aggregation.legacy.simulation.full_protocol import benchmark_scalability
from src.secure_aggregation.legacy.metrics.communication import analyze_scalability
import json


def run_scalability_experiment(
    client_counts=None,
    model_size=1000,
    output_file="results/scalability_results.json"
):
    """
    Run scalability experiments.

    Args:
        client_counts: List of client counts to test
        model_size: Model size for experiments
        output_file: Path to save results

    Returns:
        Experiment results
    """
    if client_counts is None:
        client_counts = [5, 10, 20, 50, 100]

    print(f"\n{'='*60}")
    print(f"SCALABILITY EXPERIMENT")
    print(f"{'='*60}")
    print(f"Client counts: {client_counts}")
    print(f"Model size: {model_size}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # Run full protocol benchmark
    print("Running full protocol benchmark...")
    protocol_results = benchmark_scalability(client_counts, model_size)

    # Analyze communication scalability
    print("Analyzing communication scalability...")
    comm_results = analyze_scalability(client_counts, model_size)

    # Combine results
    results = {
        'protocol_performance': protocol_results,
        'communication_scalability': comm_results,
        'config': {
            'client_counts': client_counts,
            'model_size': model_size
        }
    }

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SCALABILITY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Clients':<10} {'Time(s)':<12} {'Bytes':<15} {'Bytes/Client':<15}")
    print(f"{'-'*60}")

    for r in protocol_results['results']:
        print(f"{r['num_clients']:<10} {r['avg_time_seconds']:<12.4f} "
              f"{r['avg_bytes_sent']:<15.0f} {r['bytes_per_client']:<15.0f}")

    print(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    run_scalability_experiment()
