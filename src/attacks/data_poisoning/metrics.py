"""Attack-metric helpers for FL data poisoning experiments."""

from __future__ import annotations


def attack_success_rate(clean_accuracy: float, attacked_accuracy: float) -> float:
    """Return normalized drop in accuracy after an attack."""
    if clean_accuracy <= 0:
        return 0.0
    drop = max(clean_accuracy - attacked_accuracy, 0.0)
    return round(drop / clean_accuracy, 6)


def defense_gain(attacked_accuracy: float, defended_accuracy: float) -> float:
    """Return absolute recovery in accuracy after defense activation."""
    return round(defended_accuracy - attacked_accuracy, 6)


def summarize_attack_round(clean_accuracy: float, attacked_accuracy: float, defended_accuracy: float) -> dict[str, float]:
    """Compute the core benchmark metrics for one attack/defense run."""
    return {
        "clean_accuracy": round(clean_accuracy, 6),
        "attacked_accuracy": round(attacked_accuracy, 6),
        "defended_accuracy": round(defended_accuracy, 6),
        "attack_success_rate": attack_success_rate(clean_accuracy, attacked_accuracy),
        "defense_gain": defense_gain(attacked_accuracy, defended_accuracy),
    }
