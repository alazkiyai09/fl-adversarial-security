"""
Federated learning clients for model poisoning experiments.
"""

from .honest_client import HonestClient
from .malicious_client import MaliciousClient

__all__ = ["HonestClient", "MaliciousClient"]
