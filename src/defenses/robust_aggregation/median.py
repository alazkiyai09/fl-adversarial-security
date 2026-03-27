"""Lazy coordinate-median loader."""


def load_coordinate_median():
    from src.defenses.robust_aggregation.legacy.aggregators.median import CoordinateWiseMedian

    return CoordinateWiseMedian
