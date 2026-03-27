"""
Full secure aggregation simulation with realistic communication.

Simulates the complete protocol with network communication,
message passing, and timing.
"""

import torch
import time
from typing import List, Dict, Any, Optional
import numpy as np

from ..protocol import SecureAggregationClient, SecureAggregationServer
from ..communication import CommunicationChannel, MessageType
from ..aggregation import sum_updates
from ..utils import DropoutSimulator


def run_full_protocol_simulation(
    num_clients: int = 10,
    model_size: int = 100,
    dropout_rate: float = 0.0,
    threshold_ratio: float = 0.7,
    latency_ms: float = 10.0,
    loss_rate: float = 0.0,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run full secure aggregation simulation with realistic communication.

    Args:
        num_clients: Number of clients
        model_size: Size of model updates
        dropout_rate: Client dropout rate
        threshold_ratio: Threshold ratio for secret sharing
        latency_ms: Network latency in milliseconds
        loss_rate: Message loss rate
        seed: Random seed

    Returns:
        Dictionary with simulation results including communication costs
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"FULL SECURE AGGREGATION SIMULATION")
    print(f"{'='*60}")
    print(f"Clients: {num_clients}")
    print(f"Model size: {model_size}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Latency: {latency_ms}ms")
    print(f"Loss rate: {loss_rate}")
    print(f"{'='*60}\n")

    start_time = time.time()

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
    true_updates = []
    for _ in range(num_clients):
        update = torch.randn(model_size, dtype=torch.float32)
        true_updates.append(update)

    # Simulate dropouts
    simulator = DropoutSimulator(seed=seed)
    scenario = simulator.simulate_random(list(range(num_clients)), dropout_rate)

    active_ids = scenario.active_clients
    dead_ids = scenario.dead_ids

    print(f"Active clients: {len(active_ids)}")
    print(f"Dead clients: {len(dead_ids)}")
    print()

    # Initialize communication channel
    channel = CommunicationChannel(
        latency_ms=latency_ms,
        loss_rate=loss_rate,
        seed=seed
    )

    # Register all clients (including dead ones for simulation)
    for client_id in range(num_clients):
        channel.register_client(client_id)

    # Mark dead clients
    for dead_id in dead_ids:
        channel.unregister_client(dead_id)

    # Initialize server
    model_shape = torch.Size([model_size])
    server = SecureAggregationServer(num_clients, model_shape, config)
    server.channel = channel
    server.start_round(0)

    # Initialize active clients
    clients = {}
    for client_id in active_ids:
        client = SecureAggregationClient(
            client_id,
            true_updates[client_id],
            config
        )
        client.channel = channel
        clients[client_id] = client

    # Simulate protocol phases with communication
    phase_times = {}

    # Phase 1: Key Agreement
    phase_start = time.time()
    print("Phase 1: Key Agreement")
    all_client_ids = list(range(num_clients))
    for client_id, client in clients.items():
        client.setup_pairwise_keys(all_client_ids)
    phase_times['key_agreement'] = time.time() - phase_start

    # Phase 2: Mask Generation
    phase_start = time.time()
    print("\nPhase 2: Mask Generation")
    for client in clients.values():
        client.generate_masks_and_seeds()
        client.create_mask_shares(client.state.my_mask_seed, num_clients)
    phase_times['mask_generation'] = time.time() - phase_start

    # Phase 3: Masked Update Submission
    phase_start = time.time()
    print("\nPhase 3: Masked Update Submission")
    for client_id, client in clients.items():
        masked_update = client.submit_masked_update_final()
        server.receive_masked_update(client_id, masked_update)

        # Simulate network transmission
        channel.send(
            client_id,
            -1,  # Server
            MessageType.MASKED_UPDATE_SUBMIT,
            masked_update
        )
    phase_times['update_submission'] = time.time() - phase_start

    # Phase 4: Seed Share Submission
    phase_start = time.time()
    print("\nPhase 4: Seed Share Submission")
    for client_id, client in clients.items():
        shares = client.state.mask_shares
        server.receive_seed_shares(client_id, shares)

        # Simulate transmission
        for share in shares:
            channel.send(
                client_id,
                -1,
                MessageType.SEED_SHARE_SUBMIT,
                share
            )
    phase_times['share_submission'] = time.time() - phase_start

    # Phase 5: Dropout Recovery
    phase_start = time.time()
    if dead_ids:
        print(f"\nPhase 5: Dropout Recovery ({len(dead_ids)} dead clients)")
        from ..protocol import coordinate_recovery_protocol

        active_clients_list = list(clients.values())
        success = coordinate_recovery_protocol(server, active_clients_list, dead_ids)

        if success:
            print(f"  Successfully reconstructed {len(server.reconstructed_seeds)} seeds")
        else:
            print(f"  Recovery failed")
    else:
        print("\nPhase 5: No dropouts")
    phase_times['dropout_recovery'] = time.time() - phase_start

    # Phase 6: Aggregate Computation
    phase_start = time.time()
    print("\nPhase 6: Aggregate Computation")
    secure_aggregate = server.compute_aggregate()
    phase_times['aggregate_computation'] = time.time() - phase_start

    total_time = time.time() - start_time

    # Get communication statistics
    comm_stats = channel.get_statistics()
    server_comm_cost = server.get_communication_cost()

    print(f"\nCommunication Statistics:")
    print(f"  Messages sent: {comm_stats['total_messages_sent']}")
    print(f"  Bytes sent: {comm_stats['total_bytes_sent']}")
    print(f"  Per-client bytes: {comm_stats['total_bytes_sent'] / len(active_ids):.0f}")

    print(f"\nTiming Statistics:")
    for phase, duration in phase_times.items():
        print(f"  {phase}: {duration*1000:.2f}ms")
    print(f"  Total time: {total_time*1000:.2f}ms")

    # Verify correctness
    if secure_aggregate is not None:
        active_updates = [true_updates[i] for i in active_ids]
        expected_aggregate = sum_updates(active_updates)
        difference = torch.norm(secure_aggregate - expected_aggregate).item()

        print(f"\nVerification:")
        print(f"  L2 difference: {difference:.6e}")
        print(f"  Result: {'PASS' if difference < 1e-5 else 'FAIL'}")

        results = {
            'success': True,
            'num_clients': num_clients,
            'num_active': len(active_ids),
            'num_dead': len(dead_ids),
            'aggregate_matches': difference < 1e-5,
            'difference': difference,
            'timing': phase_times,
            'total_time': total_time,
            'communication': {
                'messages_sent': comm_stats['total_messages_sent'],
                'bytes_sent': comm_stats['total_bytes_sent'],
                'per_client_bytes': comm_stats['total_bytes_sent'] / len(active_ids)
            },
            'server_cost': server_comm_cost
        }
    else:
        results = {
            'success': False,
            'num_clients': num_clients,
            'num_active': len(active_ids),
            'num_dead': len(dead_ids),
            'error': 'Failed to compute aggregate'
        }

    print(f"\n{'='*60}\n")

    return results


def benchmark_scalability(
    client_counts: List[int],
    model_size: int = 1000,
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmark protocol scalability with varying client counts.

    Args:
        client_counts: List of client counts to test
        model_size: Size of model updates
        num_runs: Number of runs per configuration

    Returns:
        Dictionary with scalability metrics
    """
    print(f"\n{'='*60}")
    print(f"SCALABILITY BENCHMARK")
    print(f"{'='*60}\n")

    results = []

    for num_clients in client_counts:
        print(f"\nTesting {num_clients} clients...")

        run_results = []
        for run in range(num_runs):
            result = run_full_protocol_simulation(
                num_clients=num_clients,
                model_size=model_size,
                dropout_rate=0.0,  # No dropouts for scalability test
                seed=run
            )
            run_results.append(result)

        # Aggregate results
        avg_time = np.mean([r['total_time'] for r in run_results])
        avg_bytes = np.mean([r['communication']['bytes_sent'] for r in run_results])

        results.append({
            'num_clients': num_clients,
            'avg_time_seconds': avg_time,
            'avg_bytes_sent': avg_bytes,
            'bytes_per_client': avg_bytes / num_clients,
            'time_per_client_seconds': avg_time / num_clients
        })

    print(f"\n{'='*60}")
    print(f"SCALABILITY RESULTS")
    print(f"{'='*60}")
    print(f"{'Clients':<10} {'Time(s)':<12} {'Bytes':<15} {'Bytes/Client':<15}")
    print(f"{'-'*60}")

    for r in results:
        print(f"{r['num_clients']:<10} {r['avg_time_seconds']:<12.4f} "
              f"{r['avg_bytes_sent']:<15.0f} {r['bytes_per_client']:<15.0f}")

    print(f"{'='*60}\n")

    return {'results': results}
