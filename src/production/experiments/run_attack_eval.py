"""Production attack-evaluation smoke runner."""

from __future__ import annotations

from src.attacks import available_attacks
from src.production.monitoring.attack_monitor import describe_attack_monitor


def run() -> dict[str, object]:
    return {
        "experiment": "attack_eval",
        "attacks": available_attacks(),
        "monitoring": describe_attack_monitor(),
    }
