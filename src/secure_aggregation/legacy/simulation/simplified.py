"""
Simplified secure aggregation simulation.

Tests functional correctness without realistic communication overhead.
Focuses on verifying the protocol logic.
"""

import torch
from typing import List, Dict, Any, Optional
import numpy as np

from ..protocol import SecureAggregationClient, SecureAggregationServer
from ..aggregation import sum_updates
from ..utils import DropoutSimulator


def generate_mock_updates(
    num_clients: int,
    model_size: int,
    dtype: torch.dtype = torch.float32
) -> List[torch.Tensor]:
    """
    Generate mock model updates for testing.

    Args:
        num_clients: Number of clients
        model_size: Size of each update tensor
        dtype: Data type for tensors

    Returns:
        List of random update tensors
    """
    updates = []
    for _ in range(num_clients):
        update = torch.randn(model_size, dtype=dtype)
        updates.append(update)
    return updates


def run_simplified_simulation(
    num_clients: int = 10,
    model_size: int = 100,
    dropout_rate: float = 0.0,
    threshold_ratio: float = 0.7,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run simplified secure aggregation simulation.

    This focuses on functional correctness:
    - Masks cancel to zero
    - Aggregate matches plaintext sum
    - Dropout recovery works

    Args:
        num_clients: Number of clients
        model_size: Size of model updates
        dropout_rate: Client dropout rate
        threshold_ratio: Threshold ratio for secret sharing
        seed: Random seed

    Returns:
        Dictionary with simulation results
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"SIMPLIFIED SECURE AGGREGATION SIMULATION")
    print(f"{'='*60}")
    print(f"Clients: {num_clients}")
    print(f"Model size: {model_size}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Threshold ratio: {threshold_ratio}")
    print(f"{'='*60}\n")

    # Configuration
    config = {
        'num_clients': num_clients,
        'threshold_ratio': threshold_ratio,
        'dh_prime': 2**127 - 1,
        'dh_generator': 2,
        'secret_sharing_prime': 2**127 - 1,
        'dropout_tolerance': 0.3
    }

    # Generate mock updates
    true_updates = generate_mock_updates(num_clients, model_size)

    # Compute true aggregate (for verification)
    true_aggregate = sum_updates(true_updates)

    # Simulate dropouts
    simulator = DropoutSimulator(seed=seed)
    scenario = simulator.simulate_random(list(range(num_clients)), dropout_rate)

    active_ids = scenario.active_clients
    dead_ids = scenario.dead_clients

    print(f"Active clients: {len(active_ids)}")
    print(f"Dead clients: {len(dead_ids)}")
    print()

    # Initialize server
    model_shape = torch.Size([model_size])
    server = SecureAggregationServer(num_clients, model_shape, config)

    # Initialize active clients
    clients = []
    for client_id in active_ids:
        client = SecureAggregationClient(
            client_id,
            true_updates[client_id],
            config
        )
        clients.append(client)

    # Key agreement phase
    print("Phase 1: Key Agreement")
    all_client_ids = list(range(num_clients))
    for client in clients:
        client.setup_pairwise_keys(all_client_ids)
    print(f"  Completed pairwise DH for {len(clients)} clients")

    # Mask generation phase
    print("\nPhase 2: Mask Generation")
    for client in clients:
        client.generate_masks_and_seeds()
        client.create_mask_shares(client.state.my_mask_seed, num_clients)
    print(f"  Generated masks and shares for {len(clients)} clients")

    # Submit masked updates
    print("\nPhase 3: Masked Update Submission")
    masked_updates = {}
    for client in clients:
        masked_update = client.submit_masked_update_final()
        masked_updates[client.client_id] = masked_update
        server.receive_masked_update(client.client_id, masked_update)
    print(f"  Received {len(masked_updates)} masked updates")

    # Submit seed shares
    print("\nPhase 4: Seed Share Submission")
    for client in clients:
        shares = client.state.mask_shares
        server.receive_seed_shares(client.client_id, shares)
    print(f"  Received seed shares from {len(clients)} clients")

    # Handle dropouts if any
    if dead_ids:
        print(f"\nPhase 5: Dropout Recovery ({len(dead_ids)} dead clients)")
        from ..protocol import coordinate_recovery_protocol

        success = coordinate_recovery_protocol(server, clients, dead_ids)

        if success:
            print(f"  Successfully reconstructed {len(server.reconstructed_seeds)} seeds")
        else:
            print(f"  Recovery failed - insufficient shares")
    else:
        print("\nPhase 5: No dropouts - skipping recovery")

    # Compute aggregate
    print("\nPhase 6: Aggregate Computation")
    secure_aggregate = server.compute_aggregate()

    if secure_aggregate is not None:
        # Verify correctness
        # Compute aggregate from only active clients
        active_updates = [true_updates[i] for i in active_ids]
        expected_aggregate = sum_updates(active_updates)

        # Compute difference
        difference = torch.norm(secure_aggregate - expected_aggregate).item()

        print(f"  Secure aggregate computed")
        print(f"  L2 difference from expected: {difference:.6e}")

        # Verify
        is_correct = difference < 1e-5
        print(f"  Verification: {'PASS' if is_correct else 'FAIL'}")

        results = {
            'success': True,
            'num_clients': num_clients,
            'num_active': len(active_ids),
            'num_dead': len(dead_ids),
            'aggregate_matches': is_correct,
            'difference': difference,
            'true_aggregate': expected_aggregate,
            'secure_aggregate': secure_aggregate
        }
    else:
        print(f"  Failed to compute aggregate - insufficient clients")
        results = {
            'success': False,
            'num_clients': num_clients,
            'num_active': len(active_ids),
            'num_dead': len(dead_ids),
            'error': 'Insufficient clients for aggregation'
        }

    print(f"\n{'='*60}\n")

    return results


def run_batch_simulations(
    num_clients_list: List[int],
    dropout_rates: List[float],
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Run multiple simulations with varying parameters.

    Args:
        num_clients_list: List of client counts to test
        dropout_rates: List of dropout rates to test
        num_runs: Number of runs per configuration

    Returns:
        Dictionary with aggregated results
    """
    all_results = []

    for num_clients in num_clients_list:
        for dropout_rate in dropout_rates:
            print(f"\nTesting: {num_clients} clients, {dropout_rate*100:.0f}% dropout")

            for run in range(num_runs):
                result = run_simplified_simulation(
                    num_clients=num_clients,
                    dropout_rate=dropout_rate,
                    seed=run
                )
                result['run'] = run
                all_results.append(result)

    # Aggregate statistics
    success_rate = sum(1 for r in all_results if r.get('success', False)) / len(all_results)
    accuracy_stats = {
        'mean': np.mean([r.get('difference', 1.0) for r in all_results if r.get('difference') is not None]),
        'std': np.std([r.get('difference', 1.0) for r in all_results if r.get('difference') is not None])
    }

    return {
        'all_results': all_results,
        'success_rate': success_rate,
        'accuracy_stats': accuracy_stats
    }
