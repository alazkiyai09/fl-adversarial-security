"""
Flower client implementation with attack integration.
"""

from .fl_client import (
    FraudClient,
    create_client,
)

__all__ = [
    "FraudClient",
    "create_client",
]
