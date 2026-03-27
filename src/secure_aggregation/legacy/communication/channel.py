"""
Simulated communication channel for secure aggregation.

Models message passing between clients and server with configurable
latency, message loss, and client dropouts.
"""

import time
import random
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading
import queue


class MessageType(Enum):
    """Types of messages in the secure aggregation protocol."""

    # Key agreement phase
    PUBLIC_KEY_BROADCAST = "public_key_broadcast"

    # Mask submission phase
    MASKED_UPDATE_SUBMIT = "masked_update_submit"

    # Seed sharing phase
    SEED_SHARE_SUBMIT = "seed_share_submit"

    # Dropout recovery phase
    DROPOUT_NOTIFICATION = "dropout_notification"
    SHARE_REQUEST = "share_request"
    SHARE_RESPONSE = "share_response"

    # Completion
    AGGREGATE_READY = "aggregate_ready"


@dataclass
class Message:
    """A message in the system."""

    sender_id: int
    receiver_id: int  # -1 for broadcast to all
    message_type: MessageType
    payload: Any
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class CommunicationChannel:
    """
    Simulated network channel for message passing.

    Features:
    - Point-to-point and broadcast messaging
    - Configurable latency
    - Simulated message loss
    - Client dropout simulation
    """

    def __init__(
        self,
        latency_ms: float = 10.0,
        loss_rate: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the communication channel.

        Args:
            latency_ms: Average network latency in milliseconds
            loss_rate: Probability of message loss (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.latency_ms = latency_ms
        self.loss_rate = loss_rate

        if seed is not None:
            random.seed(seed)

        # Message queues for each client
        self.queues: Dict[int, queue.Queue] = {}

        # Active clients (for dropout simulation)
        self.active_clients: set = set()

        # Message logging for metrics
        self.sent_messages: List[Message] = []
        self.received_messages: Dict[int, List[Message]] = {}

        # Statistics
        self.total_bytes_sent = 0
        self.total_messages_sent = 0

    def register_client(self, client_id: int) -> None:
        """
        Register a client with the communication channel.

        Args:
            client_id: Unique client identifier
        """
        self.queues[client_id] = queue.Queue()
        self.active_clients.add(client_id)
        self.received_messages[client_id] = []

    def unregister_client(self, client_id: int) -> None:
        """
        Unregister a client (simulate dropout).

        Args:
            client_id: Client to remove
        """
        if client_id in self.active_clients:
            self.active_clients.remove(client_id)

    def send(
        self,
        sender_id: int,
        receiver_id: int,
        message_type: MessageType,
        payload: Any
    ) -> bool:
        """
        Send a message from sender to receiver.

        Args:
            sender_id: Sender's client ID
            receiver_id: Receiver's client ID (-1 for broadcast)
            message_type: Type of message
            payload: Message payload

        Returns:
            True if message sent successfully, False if dropped
        """
        # Check if sender is active
        if sender_id not in self.active_clients:
            return False

        # Simulate message loss
        if random.random() < self.loss_rate:
            return False

        # Create message
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload
        )

        # Broadcast or point-to-point
        if receiver_id == -1:
            # Broadcast to all active clients except sender
            for client_id in self.active_clients:
                if client_id != sender_id and client_id in self.queues:
                    self.queues[client_id].put(message)
                    if client_id not in self.received_messages:
                        self.received_messages[client_id] = []
                    self.received_messages[client_id].append(message)
        else:
            # Point-to-point
            if receiver_id in self.queues and receiver_id in self.active_clients:
                self.queues[receiver_id].put(message)
                if receiver_id not in self.received_messages:
                    self.received_messages[receiver_id] = []
                self.received_messages[receiver_id].append(message)
            else:
                return False  # Receiver not registered

        # Log message
        self.sent_messages.append(message)
        self.total_messages_sent += 1
        self._update_bytes_sent(payload)

        return True

    def receive(
        self,
        client_id: int,
        timeout: float = 1.0
    ) -> Optional[Message]:
        """
        Receive a message for a client.

        Args:
            client_id: Client ID
            timeout: Maximum time to wait in seconds

        Returns:
            Message if available, None if timeout
        """
        if client_id not in self.queues:
            return None

        try:
            message = self.queues[client_id].get(timeout=timeout)
            return message
        except queue.Empty:
            return None

    def receive_batch(
        self,
        client_id: int,
        count: int,
        timeout: float = 1.0
    ) -> List[Message]:
        """
        Receive multiple messages for a client.

        Args:
            client_id: Client ID
            count: Maximum number of messages to receive
            timeout: Maximum time to wait for each message

        Returns:
            List of received messages
        """
        messages = []
        for _ in range(count):
            msg = self.receive(client_id, timeout)
            if msg is not None:
                messages.append(msg)
            else:
                break
        return messages

    def _update_bytes_sent(self, payload: Any) -> None:
        """Estimate bytes sent from payload."""
        import torch

        if isinstance(payload, torch.Tensor):
            self.total_bytes_sent += payload.element_size() * payload.numel()
        elif isinstance(payload, (list, tuple)):
            self.total_bytes_sent += len(payload) * 8  # Rough estimate
        elif isinstance(payload, (int, float)):
            self.total_bytes_sent += 8
        elif isinstance(payload, dict):
            self.total_bytes_sent += len(str(payload).encode())
        else:
            self.total_bytes_sent += len(str(payload).encode())

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get communication statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'total_messages_sent': self.total_messages_sent,
            'total_bytes_sent': self.total_bytes_sent,
            'active_clients': len(self.active_clients),
            'registered_clients': len(self.queues),
            'message_log_size': len(self.sent_messages)
        }

    def reset(self) -> None:
        """Reset channel state."""
        self.queues.clear()
        self.active_clients.clear()
        self.sent_messages.clear()
        self.received_messages.clear()
        self.total_bytes_sent = 0
        self.total_messages_sent = 0
