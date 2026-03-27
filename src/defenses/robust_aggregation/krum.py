"""Lazy Krum loader."""


def load_krum():
    from src.defenses.robust_aggregation.legacy.aggregators.krum import Krum, MultiKrum

    return {"Krum": Krum, "MultiKrum": MultiKrum}
