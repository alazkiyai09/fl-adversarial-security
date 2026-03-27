"""
Security testing experiments.

Validates security properties of the protocol.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.secure_aggregation.legacy.security.verification import SecurityAuditor
from src.secure_aggregation.legacy.metrics.security import measure_security_level
import json


def run_security_tests(
    key_sizes=None,
    num_clients=10,
    output_file="results/security_analysis.json"
):
    """
    Run security tests.

    Args:
        key_sizes: List of key sizes to test
        num_clients: Number of clients
        output_file: Path to save results

    Returns:
        Security test results
    """
    if key_sizes is None:
        key_sizes = [128, 256, 512, 1024, 2048]

    print(f"\n{'='*60}")
    print(f"SECURITY ANALYSIS EXPERIMENT")
    print(f"{'='*60}")
    print(f"Key sizes: {key_sizes}")
    print(f"Clients: {num_clients}")
    print(f"{'='*60}\n")

    results = {}

    # Comprehensive security audit
    print("Running comprehensive security audit...")
    auditor = SecurityAuditor({})
    audit_results = auditor.audit_all_properties()

    results['audit'] = audit_results

    # Test different key sizes
    print("\nTesting different key sizes...")
    key_size_results = []

    for key_size in key_sizes:
        threshold = int(num_clients * 0.7)

        security = measure_security_level(
            key_size_bits=key_size,
            threshold=threshold,
            num_clients=num_clients
        )

        key_size_results.append({
            'key_size': key_size,
            'security_level': security
        })

        print(f"  {key_size} bits: {security['overall_security']}")

    results['key_size_analysis'] = key_size_results

    # Test different thresholds
    print("\nTesting different thresholds...")
    threshold_results = []

    for threshold_ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
        threshold = int(num_clients * threshold_ratio)

        security = measure_security_level(
            key_size_bits=2048,
            threshold=threshold,
            num_clients=num_clients
        )

        threshold_results.append({
            'threshold_ratio': threshold_ratio,
            'threshold': threshold,
            'security_level': security
        })

        print(f"  Threshold {threshold} ({threshold_ratio*100:.0f}%): {security['overall_security']}")

    results['threshold_analysis'] = threshold_results

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SECURITY ANALYSIS SUMMARY")
    print(f"{'='*60}")

    print(f"\nSecurity Audit:")
    print(f"  Total tests: {audit_results['total_tests']}")
    print(f"  Passed: {audit_results['passed']}")
    print(f"  Failed: {audit_results['failed']}")
    print(f"  All passed: {'YES' if audit_results['all_passed'] else 'NO'}")

    print(f"\nKey Size Analysis:")
    print(f"  {'Key Size':<10} {'Security Level':<15}")
    print(f"  {'-'*30}")
    for r in key_size_results:
        print(f"  {r['key_size']:<10} {r['security_level']['overall_security']:<15}")

    print(f"\nRecommendations:")
    for rec in key_size_results[-1]['security_level']['recommendations']:
        print(f"  • {rec}")

    print(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    run_security_tests()
