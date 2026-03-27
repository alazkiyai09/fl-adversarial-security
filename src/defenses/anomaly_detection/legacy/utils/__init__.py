"""
FL Anomaly Detection Utilities Module
"""

from .updates_parser import (
    extract_updates,
    flatten_update,
    unflatten_update,
    extract_updates_from_pytorch,
    compute_update_difference,
    batch_flatten_updates,
    get_layer_sizes,
    total_parameters
)

from .normalization import (
    normalize_by_layer_size,
    normalize_by_frobenius_norm,
    normalize_by_std_baseline,
    normalize_global,
    compute_layer_statistics,
    clip_by_norm
)

__all__ = [
    'extract_updates',
    'flatten_update',
    'unflatten_update',
    'extract_updates_from_pytorch',
    'compute_update_difference',
    'batch_flatten_updates',
    'get_layer_sizes',
    'total_parameters',
    'normalize_by_layer_size',
    'normalize_by_frobenius_norm',
    'normalize_by_std_baseline',
    'normalize_global',
    'compute_layer_statistics',
    'clip_by_norm'
]
