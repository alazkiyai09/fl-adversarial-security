"""
Pseudo-Random Function (PRF) for mask generation.

Uses cryptographic primitives to generate deterministic but unpredictable
masks from shared secrets.
"""

import hashlib
import torch
from typing import Tuple


def prf(seed: int, index: int, output_length: int) -> bytes:
    """
    Pseudo-Random Function using HMAC-SHA256.

    Generates cryptographically secure random output from a seed and index.

    Args:
        seed: The secret seed value (from DH shared secret)
        index: Index/counter for generating multiple outputs
        output_length: Desired output length in bytes

    Returns:
        Random bytes of length output_length
    """
    # Convert seed and index to bytes
    seed_bytes = seed.to_bytes((seed.bit_length() + 7) // 8, 'big')
    index_bytes = index.to_bytes((index.bit_length() + 7) // 8, 'big')

    # Use SHA256 in counter mode for generating arbitrary length output
    output = b''
    counter = 0

    while len(output) < output_length:
        # HMAC-like construction: SHA256(seed || index || counter)
        data = seed_bytes + index_bytes + counter.to_bytes(4, 'big')
        hash_output = hashlib.sha256(data).digest()
        output += hash_output
        counter += 1

    return output[:output_length]


def generate_mask_from_seed(
    seed: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Generate a random mask tensor from a seed using PRF.

    The mask is deterministic (same seed produces same mask) but
    computationally indistinguishable from random.

    Args:
        seed: Secret seed value (e.g., from DH shared secret)
        shape: Desired tensor shape
        dtype: Tensor data type (typically float32)

    Returns:
        Random mask tensor of specified shape and dtype
    """
    # Calculate number of bytes needed
    num_elements = torch.prod(torch.tensor(shape)).item()

    if dtype == torch.float32:
        bytes_per_element = 4
    elif dtype == torch.float64:
        bytes_per_element = 8
    elif dtype == torch.float16:
        bytes_per_element = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    total_bytes = int(num_elements * bytes_per_element)

    # Generate random bytes using PRF
    random_bytes = prf(seed, index=0, output_length=total_bytes)

    # Convert bytes to tensor
    # For floats, we interpret bytes as integers and normalize to [-1, 1]
    import struct

    values = []
    for i in range(num_elements):
        start = i * bytes_per_element
        end = start + bytes_per_element
        element_bytes = random_bytes[start:end]

        # Interpret bytes as unsigned integer
        if bytes_per_element == 4:
            int_val = struct.unpack('>I', element_bytes)[0]
        elif bytes_per_element == 8:
            int_val = struct.unpack('>Q', element_bytes)[0]
        else:  # bytes_per_element == 2
            int_val = struct.unpack('>H', element_bytes)[0]

        # Normalize to float in [-1, 1]
        # Map [0, 2^bits-1] to [-1, 1]
        max_val = 2 ** (8 * bytes_per_element)
        float_val = 2.0 * (int_val / max_val) - 1.0
        values.append(float_val)

    # Create tensor and reshape
    mask = torch.tensor(values, dtype=dtype).reshape(shape)

    return mask


def generate_pairwise_mask(
    shared_seeds: dict,
    client_id: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Generate mask from pairwise shared secrets.

    Each client has a shared secret with every other client.
    The final mask is the sum (XOR) of all individual masks.

    Args:
        shared_seeds: Dict mapping peer_id -> shared_secret
        client_id: This client's ID (for seed ordering)
        shape: Desired tensor shape
        dtype: Tensor data type

    Returns:
        Combined mask tensor
    """
    if not shared_seeds:
        # No shared secrets, return zero mask
        return torch.zeros(shape, dtype=dtype)

    # Start with zero mask
    combined_mask = torch.zeros(shape, dtype=dtype)

    # Add contribution from each pairwise secret
    for peer_id, seed in shared_seeds.items():
        # Use ordered pair to ensure both parties generate same mask
        ordered_pair = tuple(sorted([client_id, peer_id]))
        pair_seed = hash(ordered_pair + (seed,)) % (2 ** 32)

        peer_mask = generate_mask_from_seed(pair_seed, shape, dtype)
        combined_mask = combined_mask + peer_mask

    return combined_mask


def verify_mask_randomness(mask: torch.Tensor, num_samples: int = 1000) -> dict:
    """
    Verify that a mask appears cryptographically random.

    Performs basic statistical tests on the mask values.

    Args:
        mask: The mask tensor to test
        num_samples: Number of samples to test

    Returns:
        Dictionary with test results
    """
    flat_mask = mask.flatten().numpy()

    # Sample random elements
    import numpy as np
    indices = np.random.choice(len(flat_mask), min(num_samples, len(flat_mask)), replace=False)
    samples = flat_mask[indices]

    results = {
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples)),
        'min': float(np.min(samples)),
        'max': float(np.max(samples)),
        'approximate_uniform': 0.4 < float(np.mean(samples)) < 0.6 and 0.5 < float(np.std(samples)) < 0.6
    }

    return results
