"""Defense metadata and wrappers."""


def available_defenses() -> list[str]:
    return ["krum", "trimmed_mean", "bulyan", "median", "anomaly_detection", "foolsgold", "signguard"]
