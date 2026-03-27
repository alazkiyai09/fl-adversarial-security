"""Secure aggregation metric metadata."""


def describe_secure_aggregation_metrics() -> list[str]:
    return ["round_trip_bytes", "mask_generation_time", "dropout_recovery_success"]
