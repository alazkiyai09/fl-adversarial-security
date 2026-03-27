"""Benchmark metadata."""

from src.attacks import available_attacks
from src.defenses import available_defenses


def benchmark_surface() -> dict:
    return {
        "attacks": available_attacks(),
        "defenses": available_defenses(),
        "source": "src/benchmark/legacy",
    }
