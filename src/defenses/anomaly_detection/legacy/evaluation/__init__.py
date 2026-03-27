"""
FL Anomaly Detection Evaluation Module
"""

from .metrics import (
    compute_detection_metrics,
    compute_detection_latency,
    plot_roc_curve,
    plot_precision_recall_curve
)

__all__ = [
    'compute_detection_metrics',
    'compute_detection_latency',
    'plot_roc_curve',
    'plot_precision_recall_curve'
]
