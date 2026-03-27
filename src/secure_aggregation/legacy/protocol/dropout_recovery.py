"""
Dropout recovery protocol for secure aggregation.

Handles the scenario where clients drop out during the protocol.
Coordinates the reconstruction of masks from dropped clients.
"""

from typing import List, Dict, Tuple, Optional, Any
import torch

from .server import SecureAggregationServer
from .client import SecureAggregationClient
from ..crypto import reconstruct_secret


def validate_threshold_sufficient(
    num_clients: int,
    dead_clients: int,
    threshold: int
) -> bool:
    """
    Validate that threshold is sufficient to handle dropouts.

    We need: (num_clients - dead_clients) >= threshold
    to have enough shares for reconstruction.

    Args:
        num_clients: Total number of clients
        dead_clients: Number of dead clients
        threshold: Secret sharing threshold

    Returns:
        True if we can recover from dropouts
    """
    active_clients = num_clients - dead_clients
    return active_clients >= threshold


def coordinate_recovery_protocol(
    server: SecureAggregationServer,
    active_clients: List[SecureAggregationClient],
    dead_client_ids: List[int]
) -> bool:
    """
    Coordinate the dropout recovery protocol.

    1. Server identifies dead clients
    2. Active clients submit shares for dead clients
    3. Server reconstructs dead clients' mask seeds
    4. Masks are canceled using reconstructed seeds

    Args:
        server: The secure aggregation server
        active_clients: List of active client objects
        dead_client_ids: List of dead client IDs

    Returns:
        True if recovery succeeded, False otherwise
    """
    print(f"Starting dropout recovery for {len(dead_client_ids)} dead clients")

    # Step 1: Validate we have enough active clients
    num_active = len(active_clients)
    num_dead = len(dead_client_ids)

    if not validate_threshold_sufficient(
        num_active + num_dead,
        num_dead,
        server.threshold
    ):
        print(f"Insufficient active clients for recovery: {num_active} < {server.threshold}")
        return False

    # Step 2: Collect shares from active clients
    all_shares: Dict[int, List[Tuple[int, int]]] = {}  # dead_id -> list of shares

    for client in active_clients:
        # Get shares for dead clients
        shares = client.respond_to_dropout(dead_client_ids, [])

        for dead_id, share_idx, share_val in shares:
            if dead_id not in all_shares:
                all_shares[dead_id] = []
            all_shares[dead_id].append((share_idx, share_val))

    # Step 3: Reconstruct seeds for dead clients
    reconstructed_seeds = {}

    for dead_id, shares in all_shares.items():
        if len(shares) >= server.threshold:
            try:
                seed = reconstruct_secret(shares, server.sharing_prime)
                reconstructed_seeds[dead_id] = seed
                print(f"Reconstructed seed for client {dead_id}")
            except Exception as e:
                print(f"Failed to reconstruct seed for client {dead_id}: {e}")
                return False
        else:
            print(f"Insufficient shares for client {dead_id}: {len(shares)} < {server.threshold}")
            return False

    # Step 4: Store reconstructed seeds in server
    server.reconstructed_seeds = reconstructed_seeds

    # Step 5: Generate and cancel masks for dead clients
    from ..crypto import generate_mask_from_seed

    for dead_id, seed in reconstructed_seeds.items():
        # Reconstruct the dead client's mask
        # Note: This is simplified - real protocol needs pairwise info
        dummy_shape = server.model_shape
        mask = generate_mask_from_seed(seed, dummy_shape, torch.float32)

        # In full protocol, we'd cancel this mask
        # For now, just store it
        server.aggregator.reconstructed_masks[dead_id] = mask

    print("Dropout recovery completed successfully")
    return True


def simulate_dropouts(
    client_ids: List[int],
    dropout_rate: float,
    seed: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """
    Simulate random client dropouts.

    Args:
        client_ids: List of all client IDs
        dropout_rate: Probability of each client dropping out (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (active_client_ids, dead_client_ids)
    """
    import random

    if seed is not None:
        random.seed(seed)

    dead_clients = []
    active_clients = []

    for client_id in client_ids:
        if random.random() < dropout_rate:
            dead_clients.append(client_id)
        else:
            active_clients.append(client_id)

    return active_clients, dead_clients


def analyze_recovery_capability(
    num_clients: int,
    threshold: int,
    max_dropout_rate: float
) -> Dict[str, Any]:
    """
    Analyze the protocol's dropout recovery capability.

    Args:
        num_clients: Total number of clients
        threshold: Secret sharing threshold
        max_dropout_rate: Maximum dropout rate to analyze

    Returns:
        Dictionary with analysis results
    """
    results = {
        'num_clients': num_clients,
        'threshold': threshold,
        'max_tolerable_dropouts': num_clients - threshold,
        'max_tolerable_rate': (num_clients - threshold) / num_clients,
        'can_handle_max_rate': max_dropout_rate <= (num_clients - threshold) / num_clients,
        'breakdown': []
    }

    # Analyze various dropout rates
    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        num_dead = int(num_clients * rate)
        can_recover = validate_threshold_sufficient(num_clients, num_dead, threshold)
        results['breakdown'].append({
            'dropout_rate': rate,
            'num_dead': num_dead,
            'num_active': num_clients - num_dead,
            'can_recover': can_recover
        })

    return results


def graceful_degradation_analysis(
    num_clients: int,
    threshold: int
) -> Dict[int, str]:
    """
    Analyze how system degrades with varying numbers of dropouts.

    Args:
        num_clients: Total number of clients
        threshold: Secret sharing threshold

    Returns:
        Dictionary mapping num_dead -> status
    """
    degradation = {}

    for num_dead in range(num_clients + 1):
        num_active = num_clients - num_dead

        if num_active < threshold:
            status = "FAIL: Insufficient clients"
        elif num_dead == 0:
            status = "OPTIMAL: No dropouts"
        elif num_dead <= num_clients - threshold:
            status = "OPERATIONAL: With recovery"
        else:
            status = "FAIL: Too many dropouts"

        degradation[num_dead] = status

    return degradation
