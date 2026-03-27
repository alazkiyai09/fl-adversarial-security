"""
Security metrics for secure aggregation.

Measures the security level and privacy guarantees.
"""

from typing import Dict, List, Any
from enum import Enum


class SecurityLevel(Enum):
    """Security levels for the protocol."""

    HIGH = "high"  # 128-bit security or higher
    MEDIUM = "medium"  # 112-bit security
    LOW = "low"  # Less than 112-bit security
    INSUFFICIENT = "insufficient"  # Known vulnerabilities


def measure_security_level(
    key_size_bits: int,
    threshold: int,
    num_clients: int
) -> Dict[str, Any]:
    """
    Measure the security level of the protocol.

    Args:
        key_size_bits: Size of cryptographic keys in bits
        threshold: Secret sharing threshold
        num_clients: Total number of clients

    Returns:
        Security level assessment
    """
    # Determine security level from key size
    if key_size_bits >= 256:
        key_security = SecurityLevel.HIGH
    elif key_size_bits >= 112:
        key_security = SecurityLevel.MEDIUM
    elif key_size_bits >= 80:
        key_security = SecurityLevel.LOW
    else:
        key_security = SecurityLevel.INSUFFICIENT

    # Assess collusion resistance
    max_colluders = threshold - 1
    if max_colluders >= num_clients * 0.5:
        collusion_security = SecurityLevel.LOW
    elif max_colluders >= num_clients * 0.3:
        collusion_security = SecurityLevel.MEDIUM
    else:
        collusion_security = SecurityLevel.HIGH

    # Dropout tolerance
    dropout_tolerance = (num_clients - threshold) / num_clients
    if dropout_tolerance >= 0.3:
        dropout_grade = "good"
    elif dropout_tolerance >= 0.2:
        dropout_grade = "moderate"
    else:
        dropout_grade = "poor"

    # Overall security
    overall = min(
        [key_security, collusion_security],
        key=lambda x: list(SecurityLevel).index(x)
    )

    return {
        'key_security': key_security.value,
        'collusion_security': collusion_security.value,
        'max_colluders_resisted': max_colluders,
        'dropout_tolerance': {
            'rate': dropout_tolerance,
            'max_dead_clients': num_clients - threshold,
            'grade': dropout_grade
        },
        'overall_security': overall.value,
        'recommendations': _generate_recommendations(
            key_security,
            collusion_security,
            dropout_tolerance
        )
    }


def _generate_recommendations(
    key_security: SecurityLevel,
    collusion_security: SecurityLevel,
    dropout_tolerance: float
) -> List[str]:
    """
    Generate security recommendations.

    Args:
        key_security: Key security level
        collusion_security: Collusion security level
        dropout_tolerance: Dropout tolerance rate

    Returns:
        List of recommendation strings
    """
    recommendations = []

    if key_security != SecurityLevel.HIGH:
        recommendations.append(
            f"Increase key size to at least 256 bits (current: {key_security.value})"
        )

    if collusion_security != SecurityLevel.HIGH:
        recommendations.append(
            "Increase threshold to improve collusion resistance"
        )

    if dropout_tolerance < 0.2:
        recommendations.append(
            "Consider lowering threshold to improve dropout tolerance"
        )

    if not recommendations:
        recommendations.append("Security parameters are well-configured")

    return recommendations


def estimate_information_leakage(
    num_active_clients: int,
    threshold: int,
    num_shares_leaked: int
) -> Dict[str, Any]:
    """
    Estimate information leakage from share exposure.

    Args:
        num_active_clients: Number of active clients
        threshold: Secret sharing threshold
        num_shares_leaked: Number of shares that leaked

    Returns:
        Information leakage assessment
    """
    shares_needed = threshold - num_shares_leaked

    if num_shares_leaked >= threshold:
        leakage = "complete"
        description = "Secret can be fully reconstructed"
    elif num_shares_leaked >= threshold * 0.8:
        leakage = "high"
        description = f"Only {shares_needed} more shares needed"
    elif num_shares_leaked >= threshold * 0.5:
        leakage = "moderate"
        description = f"Significant progress: {shares_needed} shares still needed"
    elif num_shares_leaked > 0:
        leakage = "low"
        description = f"Limited exposure: {shares_needed} shares still needed"
    else:
        leakage = "none"
        description = "No shares leaked"

    return {
        'leakage_level': leakage,
        'description': description,
        'shares_leaked': num_shares_leaked,
        'shares_needed_for_reconstruction': shares_needed,
        'vulnerable': num_shares_leaked >= threshold
    }


class SecurityMetrics:
    """
    Tracks and analyzes security metrics.
    """

    def __init__(self):
        """Initialize security metrics tracker."""
        self.metrics: Dict[str, Any] = {}

    def record_key_exchange(self, key_size: int, num_pairs: int) -> None:
        """
        Record key exchange metrics.

        Args:
            key_size: Size of keys in bits
            num_pairs: Number of key pairs exchanged
        """
        self.metrics['key_exchange'] = {
            'key_size_bits': key_size,
            'num_pairs': num_pairs
        }

    def record_secret_sharing(
        self,
        threshold: int,
        num_shares: int,
        prime_bits: int
    ) -> None:
        """
        Record secret sharing metrics.

        Args:
            threshold: Threshold parameter
            num_shares: Number of shares
            prime_bits: Size of prime in bits
        """
        self.metrics['secret_sharing'] = {
            'threshold': threshold,
            'num_shares': num_shares,
            'prime_bits': prime_bits
        }

    def record_dropout(self, num_dead: int, num_total: int) -> None:
        """
        Record dropout metrics.

        Args:
            num_dead: Number of dead clients
            num_total: Total number of clients
        """
        self.metrics['dropout'] = {
            'num_dead': num_dead,
            'num_total': num_total,
            'rate': num_dead / num_total if num_total > 0 else 0
        }

    def assess_security(self) -> Dict[str, Any]:
        """
        Assess overall security based on recorded metrics.

        Returns:
            Security assessment
        """
        if not self.metrics:
            return {'status': 'insufficient_data'}

        key_size = self.metrics.get('key_exchange', {}).get('key_size_bits', 0)
        threshold = self.metrics.get('secret_sharing', {}).get('threshold', 0)
        num_total = self.metrics.get('dropout', {}).get('num_total', 0)

        return measure_security_level(
            key_size_bits=key_size,
            threshold=threshold,
            num_clients=num_total
        )

    def reset(self) -> None:
        """Reset metrics."""
        self.metrics.clear()
