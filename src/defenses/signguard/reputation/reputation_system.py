"""Educational SignGuard reputation metadata."""


def describe_reputation_system() -> dict:
    return {"source": "src/defenses/signguard/legacy/reputation", "policy": "time_decay_ema"}
