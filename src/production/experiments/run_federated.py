"""Production FL smoke experiment runner."""

from __future__ import annotations

from src.production.fl.server import describe_production_server


def run() -> dict[str, object]:
    return {
        "experiment": "production_federated",
        "server": describe_production_server(),
    }
