"""Federated Learning module using Flower framework."""

from .server import FlowerServer
from .client import FlowerClient
from .strategy import create_strategy, get_strategy_config
from .defenses import SignGuardDefense

__all__ = [
    "FlowerServer",
    "FlowerClient",
    "create_strategy",
    "get_strategy_config",
    "SignGuardDefense",
]
