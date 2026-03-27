"""
Flower server implementation with defense integration.
"""

from .fl_server import (
    DefendedFedAvg,
    create_server,
    ServerConfig,
)

__all__ = [
    "DefendedFedAvg",
    "create_server",
    "ServerConfig",
]
