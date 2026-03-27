"""
Dropout analysis experiments.

Analyzes protocol behavior under various dropout scenarios.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.secure_aggregation.legacy.simulation.simplified import run_simplified_simulation
from src.secure_aggregation.legacy.protocol.dropout_recovery import graceful_degradation_analysis
import json


def run_dropout_analysis(
    num_clients=10,
    model_size=100,
    dropout_rates=None,
    output_file="results/dropout_analysis.json"
):
    """
    Run dropout analysis experiments.

    Args:
        num_clients: Number of clients
        model_size: Model update size
        dropout_rates: List of dropout rates to test
        output_file: Path to save results

    Returns:
        Experiment results
    """
    if dropout_rates is None:
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"\n{'='*60}")
    print(f"DROPOUT ANALYSIS EXPERIMENT")
    print(f"{'='*60}")
    print(f"Clients: {num_clients}")
    print(f"Model size: {model_size}")
    print(f"Dropout rates: {dropout_rates}")
    print(f"{'='*60}\n")

    results = []

    for dropout_rate in dropout_rates:
        print(f"\nTesting {dropout_rate*100:.0f}% dropout...")

        # Run multiple trials
        trial_results = []
        for trial in range(5):
            result = run_simplified_simulation(
                num_clients=num_clients,
                model_size=model_size,
                dropout_rate=dropout_rate,
                seed=trial
            )
            trial_results.append(result)

        # Aggregate statistics
        success_count = sum(1 for r in trial_results if r['success'])
        avg_difference = (
            sum(r['difference'] for r in trial_results if r.get('difference'))
            / len([r for r in trial_results if r.get('difference')])
            if any(r.get('difference') for r in trial_results)
            else None
        )

        results.append({
            'dropout_rate': dropout_rate,
            'success_rate': success_count / len(trial_results),
            'avg_difference': avg_difference,
            'trials': trial_results
        })

        print(f"  Success rate: {success_count}/{len(trial_results)}")

    # Graceful degradation analysis
    print("\nGraceful degradation analysis:")
    threshold = int(num_clients * 0.7)
    degradation = graceful_degradation_analysis(num_clients, threshold)

    # Save results
    output = {
        'config': {
            'num_clients': num_clients,
            'model_size': model_size,
            'threshold': threshold
        },
        'results': results,
        'graceful_degradation': degradation
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"DROPOUT ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dropout':<10} {'Success Rate':<15} {'Avg Diff':<15}")
    print(f"{'-'*60}")

    for r in results:
        print(f"{r['dropout_rate']*100:>6.0f}%    "
              f"{r['success_rate']*100:>10.0f}%      "
              f"{r['avg_difference'] or 'N/A':<12}")

    print(f"{'='*60}\n")

    return output


if __name__ == '__main__':
    run_dropout_analysis()
