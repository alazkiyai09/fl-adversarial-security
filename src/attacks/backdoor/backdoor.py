"""Backdoor wrapper metadata."""


def describe_backdoor_attack() -> dict:
    return {"source": "src/attacks/backdoor/legacy", "triggers": ["pattern", "semantic", "distributed"]}
