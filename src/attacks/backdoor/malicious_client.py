"""Lazy malicious-client loader."""


def load_malicious_client():
    from src.attacks.backdoor.legacy.clients.malicious_client import MaliciousClient

    return MaliciousClient
