"""
Communication overhead metrics for secure aggregation.

Measures and analyzes the additional bandwidth required by
secure aggregation compared to plaintext aggregation.
"""

import torch
from typing import Dict, List, Any, Tuple
import numpy as np


def get_tensor_bytes(tensor: torch.Tensor) -> int:
    """Get the size of a tensor in bytes."""
    return tensor.element_size() * tensor.numel()


def measure_communication_overhead(
    num_clients: int,
    model_size: int,
    threshold: int,
    include_dropout_recovery: bool = False
) -> Dict[str, Any]:
    """
    Measure the communication overhead of secure aggregation.

    Args:
        num_clients: Number of clients
        model_size: Size of model update tensors
        threshold: Secret sharing threshold
        include_dropout_recovery: Whether to include dropout recovery messages

    Returns:
        Dictionary with overhead breakdown
    """
    dtype = torch.float32
    bytes_per_param = 4  # float32

    # Plaintext baseline
    plaintext_bytes_per_client = model_size * bytes_per_param
    plaintext_total = num_clients * plaintext_bytes_per_client

    # Secure aggregation components
    # 1. Key exchange: public keys (2048 bits = 256 bytes)
    key_exchange_bytes = num_clients * 256

    # 2. Masked updates: same size as plaintext
    masked_update_bytes = plaintext_total

    # 3. Seed shares: each client creates n shares
    # Each share is a large prime (128 bits = 16 bytes)
    shares_per_client = num_clients
    bytes_per_share = 16
    seed_share_bytes = num_clients * shares_per_client * bytes_per_share

    # Total
    secure_total = key_exchange_bytes + masked_update_bytes + seed_share_bytes

    # Dropout recovery overhead
    recovery_bytes = 0
    if include_dropout_recovery:
        # Assume 30% dropout rate
        num_dead = int(num_clients * 0.3)
        # Each active client sends shares for dead clients
        num_active = num_clients - num_dead
        recovery_bytes = num_active * num_dead * bytes_per_share

    total_with_recovery = secure_total + recovery_bytes

    # Calculate overhead
    overhead_ratio = total_with_recovery / plaintext_total if plaintext_total > 0 else 0
    additional_bytes = total_with_recovery - plaintext_total

    return {
        'plaintext_baseline': {
            'per_client_bytes': plaintext_bytes_per_client,
            'total_bytes': plaintext_total
        },
        'secure_aggregation': {
            'key_exchange_bytes': key_exchange_bytes,
            'masked_update_bytes': masked_update_bytes,
            'seed_share_bytes': seed_share_bytes,
            'recovery_bytes': recovery_bytes,
            'total_bytes': total_with_recovery
        },
        'overhead': {
            'ratio': overhead_ratio,
            'additional_bytes': additional_bytes,
            'percentage': (overhead_ratio - 1) * 100
        },
        'breakdown_per_client': {
            'bytes_sent': total_with_recovery / num_clients,
            'key_exchange': 256,
            'masked_update': plaintext_bytes_per_client,
            'seed_shares': shares_per_client * bytes_per_share,
            'recovery': recovery_bytes / num_clients if num_clients > 0 else 0
        }
    }


def compare_with_plaintext(
    model_sizes: List[int],
    num_clients: int = 10
) -> Dict[str, Any]:
    """
    Compare secure aggregation with plaintext across model sizes.

    Args:
        model_sizes: List of model sizes to test
        num_clients: Number of clients

    Returns:
        Comparison results
    """
    results = []

    for model_size in model_sizes:
        plaintext = model_size * 4 * num_clients  # float32

        overhead = measure_communication_overhead(
            num_clients=num_clients,
            model_size=model_size,
            threshold=int(num_clients * 0.7)
        )

        results.append({
            'model_size': model_size,
            'plaintext_bytes': plaintext,
            'secure_bytes': overhead['secure_aggregation']['total_bytes'],
            'overhead_ratio': overhead['overhead']['ratio'],
            'overhead_percentage': overhead['overhead']['percentage']
        })

    return {
        'model_sizes': model_sizes,
        'num_clients': num_clients,
        'results': results
    }


def analyze_scalability(
    client_counts: List[int],
    model_size: int = 1000
) -> Dict[str, Any]:
    """
    Analyze how communication scales with number of clients.

    Args:
        client_counts: List of client counts to test
        model_size: Size of model updates

    Returns:
        Scalability analysis
    """
    results = []

    for num_clients in client_counts:
        overhead = measure_communication_overhead(
            num_clients=num_clients,
            model_size=model_size,
            threshold=int(num_clients * 0.7)
        )

        results.append({
            'num_clients': num_clients,
            'total_bytes': overhead['secure_aggregation']['total_bytes'],
            'per_client_bytes': overhead['breakdown_per_client']['bytes_sent'],
            'overhead_ratio': overhead['overhead']['ratio']
        })

    return {
        'model_size': model_size,
        'results': results
    }


class CommunicationProfiler:
    """
    Profiles communication patterns in secure aggregation.
    """

    def __init__(self):
        """Initialize the profiler."""
        self.message_log: List[Dict[str, Any]] = []
        self.total_bytes = 0

    def log_message(
        self,
        sender: int,
        receiver: int,
        message_type: str,
        size_bytes: int
    ) -> None:
        """
        Log a message for profiling.

        Args:
            sender: Sender ID
            receiver: Receiver ID
            message_type: Type of message
            size_bytes: Size in bytes
        """
        self.message_log.append({
            'sender': sender,
            'receiver': receiver,
            'type': message_type,
            'size_bytes': size_bytes
        })
        self.total_bytes += size_bytes

    def get_summary(self) -> Dict[str, Any]:
        """
        Get communication summary.

        Returns:
            Summary statistics
        """
        if not self.message_log:
            return {
                'total_messages': 0,
                'total_bytes': 0,
                'average_message_size': 0
            }

        # Group by message type
        by_type: Dict[str, List[Dict]] = {}
        for msg in self.message_log:
            msg_type = msg['type']
            if msg_type not in by_type:
                by_type[msg_type] = []
            by_type[msg_type].append(msg)

        type_stats = {}
        for msg_type, messages in by_type.items():
            total_bytes = sum(m['size_bytes'] for m in messages)
            type_stats[msg_type] = {
                'count': len(messages),
                'total_bytes': total_bytes,
                'average_bytes': total_bytes / len(messages)
            }

        return {
            'total_messages': len(self.message_log),
            'total_bytes': self.total_bytes,
            'average_message_size': self.total_bytes / len(self.message_log),
            'by_type': type_stats
        }

    def reset(self) -> None:
        """Reset the profiler."""
        self.message_log.clear()
        self.total_bytes = 0
