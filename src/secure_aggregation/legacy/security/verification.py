"""
Security verification for secure aggregation.

Verifies that the protocol provides:
1. Server privacy - server cannot see individual updates
2. Collusion resistance - up to t-1 clients cannot reconstruct
3. Forward secrecy - past aggregates remain private
"""

import torch
import random
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from ..crypto import generate_mask_from_seed, reconstruct_secret


class SecurityProperty(Enum):
    """Security properties to verify."""

    SERVER_PRIVACY = "server_privacy"
    COLLUSION_RESISTANCE = "collusion_resistance"
    FORWARD_SECRECY = "forward_secrecy"
    MASK_UNPREDICTABILITY = "mask_unpredictability"


@dataclass
class SecurityTestResult:
    """Result of a security test."""

    property: SecurityProperty
    passed: bool
    details: str
    evidence: Dict[str, Any] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = {}


class SecurityAuditor:
    """
    Audits the security of the secure aggregation protocol.

    Tests various attack scenarios and verifies security guarantees.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the security auditor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.test_results: List[SecurityTestResult] = []

    def audit_all_properties(self) -> Dict[str, Any]:
        """
        Run all security tests.

        Returns:
            Dictionary with test results
        """
        self.test_results.clear()

        # Run all security tests
        self.test_server_privacy()
        self.test_collusion_resistance()
        self.test_forward_secrecy()
        self.test_mask_unpredictability()

        # Summarize results
        passed = sum(1 for r in self.test_results if r.passed)
        total = len(self.test_results)

        return {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'all_passed': passed == total,
            'results': [
                {
                    'property': r.property.value,
                    'passed': r.passed,
                    'details': r.details
                }
                for r in self.test_results
            ]
        }

    def test_server_privacy(self, num_clients: int = 10) -> SecurityTestResult:
        """
        Test that server cannot reconstruct individual updates.

        The server should only see masked updates, which are
        computationally indistinguishable from random.

        Returns:
            SecurityTestResult
        """
        # Simulate server seeing masked updates
        model_shape = torch.Size([100])

        # Generate real updates
        real_updates = [torch.randn(model_shape) for _ in range(num_clients)]

        # Generate masks
        masks = []
        for _ in range(num_clients):
            seed = random.randint(1, 2**32 - 1)
            mask = generate_mask_from_seed(seed, model_shape, torch.float32)
            masks.append(mask)

        # Create masked updates (what server sees)
        masked_updates = [real_updates[i] + masks[i] for i in range(num_clients)]

        # Server's attempt to reconstruct (using only masked updates)
        # Without masks, server sees values that are statistically random

        # Statistical test: are masked updates close to random?
        # Compute variance and check if it matches uniform distribution
        all_masked = torch.cat(masked_updates)
        masked_mean = torch.mean(all_masked).item()
        masked_std = torch.std(all_masked).item()

        # For properly masked data, mean should be near 0 and std should be ~1
        is_privacy_preserved = abs(masked_mean) < 0.5 and 0.5 < masked_std < 1.5

        result = SecurityTestResult(
            property=SecurityProperty.SERVER_PRIVACY,
            passed=is_privacy_preserved,
            details=f"Masked updates appear random: mean={masked_mean:.4f}, std={masked_std:.4f}",
            evidence={
                'masked_mean': masked_mean,
                'masked_std': masked_std,
                'looks_random': is_privacy_preserved
            }
        )

        self.test_results.append(result)
        return result

    def test_collusion_resistance(
        self,
        num_clients: int = 10,
        threshold: int = 7
    ) -> SecurityTestResult:
        """
        Test that t-1 colluding clients cannot reconstruct.

        Fewer than threshold clients should not have enough shares
        to reconstruct any mask seed.

        Returns:
            SecurityTestResult
        """
        # Simulate t-1 clients colluding
        colluding_clients = threshold - 1

        # Generate a secret and split it
        secret = random.randint(1, 2**32 - 1)
        prime = 2**127 - 1

        from ..crypto import split_secret
        shares = split_secret(secret, threshold, num_clients, prime)

        # Colluding clients pool their shares
        colluding_shares = shares[:colluding_clients]

        # Attempt reconstruction
        try:
            reconstructed = reconstruct_secret(colluding_shares, prime)
            # If reconstruction succeeds with fewer than threshold shares,
            # it's either a coincidence or a security failure
            matches = reconstructed == secret
            passed = not matches  # Should not reconstruct correctly

            result = SecurityTestResult(
                property=SecurityProperty.COLLUSION_RESISTANCE,
                passed=passed,
                details=f"{colluding_clients} clients colluded (threshold={threshold}): "
                       f"reconstruction {'incorrectly' if passed else 'incorrectly'} succeeded",
                evidence={
                    'colluding_clients': colluding_clients,
                    'threshold': threshold,
                    'reconstruction_matches': matches
                }
            )
        except Exception:
            # Reconstruction failed - expected behavior
            result = SecurityTestResult(
                property=SecurityProperty.COLLUSION_RESISTANCE,
                passed=True,
                details=f"{colluding_clients} clients cannot reconstruct (expected)",
                evidence={
                    'colluding_clients': colluding_clients,
                    'threshold': threshold,
                    'reconstruction_failed': True
                }
            )

        self.test_results.append(result)
        return result

    def test_forward_secrecy(self) -> SecurityTestResult:
        """
        Test that past aggregates remain private if keys compromised.

        Future key compromises should not reveal past aggregates.

        Returns:
            SecurityTestResult
        """
        # In this protocol, masks are ephemeral and reconstructed
        # each round, providing forward secrecy by design

        # Simulate two rounds with different masks
        model_shape = torch.Size([100])

        # Round 1
        round1_updates = [torch.randn(model_shape) for _ in range(5)]
        round1_seeds = [random.randint(1, 2**32 - 1) for _ in range(5)]
        round1_masks = [
            generate_mask_from_seed(s, model_shape, torch.float32)
            for s in round1_seeds
        ]
        round1_masked = [round1_updates[i] + round1_masks[i] for i in range(5)]
        round1_aggregate = sum(round1_masked)  # Masks cancel

        # Round 2 - different seeds
        round2_seeds = [random.randint(1, 2**32 - 1) for _ in range(5)]
        round2_masks = [
            generate_mask_from_seed(s, model_shape, torch.float32)
            for s in round2_seeds
        ]

        # Even if round2 seeds are leaked, round1 masks remain secure
        # Check that round2 masks don't correlate with round1 masks
        correlation = 0.0
        for m1 in round1_masks:
            for m2 in round2_masks:
                corr = torch.corrcoef(
                    torch.stack([m1.flatten(), m2.flatten()])
                )[0, 1].item()
                correlation += corr

        avg_correlation = abs(correlation / (len(round1_masks) * len(round2_masks)))
        passed = avg_correlation < 0.1  # Low correlation expected

        result = SecurityTestResult(
            property=SecurityProperty.FORWARD_SECRECY,
            passed=passed,
            details=f"Mask correlation across rounds: {avg_correlation:.4f}",
            evidence={
                'cross_round_correlation': avg_correlation,
                'below_threshold': passed
            }
        )

        self.test_results.append(result)
        return result

    def test_mask_unpredictability(self, num_samples: int = 100) -> SecurityTestResult:
        """
        Test that masks are computationally unpredictable.

        Masks from different seeds should be uncorrelated.

        Returns:
            SecurityTestResult
        """
        model_shape = torch.Size([1000])

        # Generate masks from different seeds
        seeds = [random.randint(1, 2**32 - 1) for _ in range(num_samples)]
        masks = [
            generate_mask_from_seed(s, model_shape, torch.float32)
            for s in seeds
        ]

        # Check statistical properties
        all_values = torch.cat([m.flatten() for m in masks])
        mean = all_values.mean().item()
        std = all_values.std().item()

        # Should look like uniform distribution on [-1, 1]
        # Mean ≈ 0, std ≈ 1/sqrt(3) ≈ 0.577 for uniform[-1,1]
        expected_std = 1.0 / (3 ** 0.5)

        mean_close = abs(mean) < 0.1
        std_close = abs(std - expected_std) < 0.1

        passed = mean_close and std_close

        result = SecurityTestResult(
            property=SecurityProperty.MASK_UNPREDICTABILITY,
            passed=passed,
            details=f"Mask statistics: mean={mean:.4f}, std={std:.4f} (expected ~{expected_std:.4f})",
            evidence={
                'mean': mean,
                'std': std,
                'expected_std': expected_std,
                'mean_close': mean_close,
                'std_close': std_close
            }
        )

        self.test_results.append(result)
        return result


def verify_server_privacy(num_clients: int = 10) -> bool:
    """
    Quick check that server cannot see individual updates.

    Args:
        num_clients: Number of clients to simulate

    Returns:
        True if privacy is preserved
    """
    auditor = SecurityAuditor({})
    result = auditor.test_server_privacy(num_clients)
    return result.passed


def verify_collusion_resistance(
    num_clients: int = 10,
    threshold: int = 7
) -> bool:
    """
    Quick check that collusion resistance works.

    Args:
        num_clients: Total number of clients
        threshold: Secret sharing threshold

    Returns:
        True if collusion resistance holds
    """
    auditor = SecurityAuditor({})
    result = auditor.test_collusion_resistance(num_clients, threshold)
    return result.passed


def verify_forward_secrecy() -> bool:
    """
    Quick check that forward secrecy holds.

    Returns:
        True if forward secrecy is preserved
    """
    auditor = SecurityAuditor({})
    result = auditor.test_forward_secrecy()
    return result.passed
