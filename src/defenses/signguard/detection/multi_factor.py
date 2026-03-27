"""Educational SignGuard detection metadata."""


def describe_multi_factor_detection() -> dict:
    return {"source": "src/defenses/signguard/legacy/detection", "signals": ["magnitude", "direction", "loss"]}
