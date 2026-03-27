"""Lazy trimmed-mean loader."""


def load_trimmed_mean():
    from src.defenses.robust_aggregation.legacy.aggregators.trimmed_mean import TrimmedMean

    return TrimmedMean
