"""Attack metadata and wrappers."""


def available_attacks() -> list[str]:
    return [
        "label_flipping",
        "backdoor",
        "model_poisoning",
        "gradient_leakage",
        "membership_inference",
        "property_inference",
    ]
