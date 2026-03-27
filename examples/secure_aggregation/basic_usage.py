"""
Basic usage example for secure aggregation.

Demonstrates the simplest way to use secure aggregation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.simplified import run_simplified_simulation


def main():
    """Run basic secure aggregation example."""
    print("="*60)
    print("SECURE AGGREGATION - BASIC EXAMPLE")
    print("="*60)
    print()

    # Run a simple simulation with 10 clients
    result = run_simplified_simulation(
        num_clients=10,
        model_size=100,
        dropout_rate=0.1,
        seed=42
    )

    # Print results
    print("\nRESULTS:")
    print(f"  Success: {result['success']}")
    print(f"  Active clients: {result['num_active']}")
    print(f"  Dead clients: {result['num_dead']}")
    print(f"  Aggregate matches: {result['aggregate_matches']}")
    print(f"  L2 difference: {result['difference']:.6e}")

    if result['aggregate_matches']:
        print("\n✓ Secure aggregation working correctly!")
    else:
        print("\n✗ Aggregate verification failed")


if __name__ == '__main__':
    main()
