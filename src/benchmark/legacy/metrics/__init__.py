"""
Metrics and statistical analysis for attack/defense evaluation.
"""

from .attack_metrics import (
    compute_attack_success_rate,
    compute_auprc,
    compute_clean_accuracy,
    compute_fraud_detection_metrics,
    compute_model_size,
    MetricsHistory,
)

from .statistical_tests import (
    paired_t_test,
    wilcoxon_signed_rank_test,
    compute_effect_size,
    multiple_comparison_correction,
    anova_test,
    compute_confidence_interval,
    bootstrap_confidence_interval,
    summarize_results,
)

__all__ = [
    "compute_attack_success_rate",
    "compute_auprc",
    "compute_clean_accuracy",
    "compute_fraud_detection_metrics",
    "compute_model_size",
    "MetricsHistory",
    "paired_t_test",
    "wilcoxon_signed_rank_test",
    "compute_effect_size",
    "multiple_comparison_correction",
    "anova_test",
    "compute_confidence_interval",
    "bootstrap_confidence_interval",
    "summarize_results",
]
