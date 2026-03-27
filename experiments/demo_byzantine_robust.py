#!/usr/bin/env python3
"""
Quick demo script showing Byzantine-robust aggregator usage.
"""

import torch
from src.aggregators import (
    CoordinateWiseMedian,
    TrimmedMean,
    Krum,
    MultiKrum,
    Bulyan
)


def create_mock_updates(n_clients=10, n_attackers=2):
    """Create mock client updates with some Byzantine clients."""
    updates = []

    for i in range(n_clients):
        if i < n_attackers:
            # Malicious update (wrong direction)
            update = {
                'layer1.weight': torch.randn(10, 5) * 10 + 100,  # Far from honest
                'layer1.bias': torch.randn(10) * 5 + 50,
            }
        else:
            # Honest update (small random gradients)
            update = {
                'layer1.weight': torch.randn(10, 5) * 0.1,  # Small, reasonable
                'layer1.bias': torch.randn(10) * 0.1,
            }
        updates.append(update)

    return updates


def main():
    print("=" * 70)
    print("Byzantine-Robust Aggregator Demo")
    print("=" * 70)

    # Create mock scenario
    n_clients = 10
    n_attackers = 2

    print(f"\nScenario: {n_clients} clients, {n_attackers} attackers ({n_attackers/n_clients:.0%})")
    print("-" * 70)

    updates = create_mock_updates(n_clients, n_attackers)

    # Test each aggregator
    aggregators = [
        ('Median', CoordinateWiseMedian()),
        ('TrimmedMean', TrimmedMean(beta=0.2)),
        ('Krum', Krum()),
        ('MultiKrum', MultiKrum(m=5)),
        ('Bulyan', Bulyan()),
    ]

    for name, agg in aggregators:
        try:
            result = agg.aggregate(updates, num_attackers=n_attackers)
            print(f"\n{name:15s}: SUCCESS")
            print(f"  - Aggregated 'layer1.weight' shape: {result['layer1.weight'].shape}")
            print(f"  - Aggregated 'layer1.weight' mean: {result['layer1.weight'].mean().item():.4f}")
            print(f"  - Aggregated 'layer1.bias' mean: {result['layer1.bias'].mean().item():.4f}")
        except Exception as e:
            print(f"\n{name:15s}: FAILED - {e}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
