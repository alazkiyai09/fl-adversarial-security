"""Evaluation metrics and comparison utilities for robust aggregators."""

from .metrics import (
    compute_accuracy,
    compute_attack_success_rate,
    compute_convergence_speed
)
from .comparison import generate_comparison_matrix, generate_heatmap

__all__ = [
    'compute_accuracy',
    'compute_attack_success_rate',
    'compute_convergence_speed',
    'generate_comparison_matrix',
    'generate_heatmap',
]
