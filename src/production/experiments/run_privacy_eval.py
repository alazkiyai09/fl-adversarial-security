"""Production privacy-evaluation smoke runner."""

from __future__ import annotations

from src.production.privacy.dp import describe_dp_layer
from src.production.privacy.secure_agg import describe_secure_agg_layer


def run() -> dict[str, object]:
    return {
        "experiment": "privacy_eval",
        "dp": describe_dp_layer(),
        "secure_aggregation": describe_secure_agg_layer(),
    }
