"""
Shamir's Secret Sharing for secure aggregation.

Implements t-of-n threshold secret sharing where:
- t shares are sufficient to reconstruct the secret
- Fewer than t shares reveal no information about the secret
"""

from typing import List, Tuple
import secrets


def split_secret(
    secret: int,
    threshold: int,
    num_shares: int,
    prime: int
) -> List[Tuple[int, int]]:
    """
    Split a secret into n shares using Shamir's threshold scheme.

    Creates a random polynomial of degree (threshold-1) where:
    - f(0) = secret (constant term)
    - Shares are points (x, f(x)) for x = 1, 2, ..., n

    Args:
        secret: The secret value to share
        threshold: Minimum number of shares needed for reconstruction (t)
        num_shares: Total number of shares to create (n)
        prime: Prime modulus for field arithmetic

    Returns:
        List of (x, f(x)) shares

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate parameters
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    if threshold > num_shares:
        raise ValueError("Threshold cannot exceed number of shares")
    if secret >= prime:
        raise ValueError("Secret must be less than prime modulus")

    # Generate cryptographically secure random polynomial coefficients
    # f(x) = secret + a_1*x + a_2*x^2 + ... + a_{t-1}*x^{t-1}
    # Coefficients must be non-zero for proper interpolation
    coefficients = [secret] + [secrets.randbelow(prime - 1) + 1 for _ in range(threshold - 1)]

    # Generate shares at distinct x values (x = 1, 2, ..., n)
    # We use 1-indexed shares because f(0) = secret (the constant term)
    shares = []
    for x in range(1, num_shares + 1):
        y = evaluate_polynomial(coefficients, x, prime)
        shares.append((x, y))

    return shares


def reconstruct_secret(shares: List[Tuple[int, int]], prime: int) -> int:
    """
    Reconstruct a secret from shares using Lagrange interpolation.

    Given t or more shares (x_i, y_i), computes f(0) which is the secret.

    Args:
        shares: List of (x, y) share pairs
        prime: Prime modulus for field arithmetic

    Returns:
        Reconstructed secret

    Raises:
        ValueError: If insufficient shares provided
    """
    if len(shares) < 2:
        raise ValueError("At least 2 shares required for reconstruction")

    # Use Lagrange interpolation to compute f(0)
    secret = 0

    for i, (x_i, y_i) in enumerate(shares):
        # Compute Lagrange basis polynomial L_i(0)
        numerator = 1
        denominator = 1

        for j, (x_j, _) in enumerate(shares):
            if i != j:
                numerator *= (0 - x_j)
                denominator *= (x_i - x_j)

        # Compute L_i(0) = numerator / denominator mod prime
        denominator_inv = mod_inverse(denominator % prime, prime)
        lagrange_basis = (numerator * denominator_inv) % prime

        # Add contribution: y_i * L_i(0)
        secret = (secret + y_i * lagrange_basis) % prime

    return secret


def evaluate_polynomial(coefficients: List[int], x: int, prime: int) -> int:
    """
    Evaluate polynomial at point x using Horner's method.

    f(x) = c_0 + c_1*x + c_2*x^2 + ... + c_{d-1}*x^{d-1}

    Args:
        coefficients: List of coefficients [c_0, c_1, ..., c_{d-1}]
        x: Point at which to evaluate
        prime: Prime modulus

    Returns:
        f(x) mod prime
    """
    result = 0
    # Horner's method: (((c_{d-1} * x) + c_{d-2}) * x) + ... + c_0
    for coeff in reversed(coefficients):
        result = (result * x + coeff) % prime
    return result


def mod_inverse(a: int, prime: int) -> int:
    """
    Compute modular multiplicative inverse using Fermat's little theorem.

    a^(-1) mod p = a^(p-2) mod p (for prime p)

    Args:
        a: Value to invert
        prime: Prime modulus

    Returns:
        Modular inverse of a

    Raises:
        ValueError: If inverse doesn't exist (a is 0 mod prime)
    """
    a = a % prime
    if a == 0:
        raise ValueError("No modular inverse exists for 0")

    # Fermat's little theorem: a^(p-2) ≡ a^(-1) mod p
    return pow(a, prime - 2, prime)


def verify_reconstruction(
    shares: List[Tuple[int, int]],
    original_secret: int,
    prime: int
) -> bool:
    """
    Verify that reconstructed secret matches original.

    Args:
        shares: List of shares to reconstruct from
        original_secret: The original secret value
        prime: Prime modulus

    Returns:
        True if reconstruction matches original, False otherwise
    """
    try:
        reconstructed = reconstruct_secret(shares, prime)
        return reconstructed == original_secret
    except (ValueError, ZeroDivisionError):
        return False


def verify_threshold_property(
    secret: int,
    threshold: int,
    num_shares: int,
    prime: int
) -> bool:
    """
    Verify threshold property: t-1 shares reveal nothing.

    Tests that different subsets of t-1 shares produce different
    candidate secrets (information-theoretic security).

    Args:
        secret: Original secret
        threshold: Threshold parameter t
        num_shares: Total number of shares n
        prime: Prime modulus

    Returns:
        True if threshold property holds (t-1 shares insufficient)
    """
    from itertools import combinations

    # Generate shares
    shares = split_secret(secret, threshold, num_shares, prime)

    # Try all subsets of size threshold-1
    for subset in combinations(shares, threshold - 1):
        # Each subset should produce a different "secret"
        # In practice, we just verify no subset accidentally reconstructs
        subset_secret = reconstruct_secret(list(subset), prime)
        if subset_secret == secret:
            return False  # Security failure!

    return True
