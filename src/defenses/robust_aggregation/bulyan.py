"""Lazy Bulyan loader."""


def load_bulyan():
    from src.defenses.robust_aggregation.legacy.aggregators.bulyan import Bulyan

    return Bulyan
