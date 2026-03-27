"""
Example demonstrating dropout recovery in secure aggregation.

Shows how the protocol handles client dropouts.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.simplified import run_simplified_simulation


def test_dropout_scenarios():
    """Test various dropout scenarios."""
    print("="*60)
    print("DROPOUT RECOVERY SCENARIOS")
    print("="*60)
    print()

    scenarios = [
        (0.0, "No dropouts"),
        (0.1, "10% dropout"),
        (0.2, "20% dropout"),
        (0.3, "30% dropout (max tolerance)"),
        (0.4, "40% dropout (should fail)")
    ]

    results = []

    for dropout_rate, description in scenarios:
        print(f"\nTesting: {description}")
        print("-" * 60)

        result = run_simplified_simulation(
            num_clients=10,
            model_size=100,
            dropout_rate=dropout_rate,
            seed=42
        )

        success = result['success']
        active = result['num_active']
        dead = result['num_dead']

        print(f"  Active: {active}, Dead: {dead}")
        print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")

        results.append({
            'dropout_rate': dropout_rate,
            'description': description,
            'success': success,
            'active': active,
            'dead': dead
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = sum(1 for r in results if r['success'])
    total = len(results)

    print(f"\nSuccessful scenarios: {successful}/{total}")

    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['description']}: {r['active']} active, {r['dead']} dead")


if __name__ == '__main__':
    test_dropout_scenarios()
