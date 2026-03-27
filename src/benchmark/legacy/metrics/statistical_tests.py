"""
Statistical analysis for comparing attack and defense performance.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from scipy import stats


def paired_t_test(
    results_a: np.ndarray,
    results_b: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Perform paired t-test on two sets of results.

    Tests whether the mean difference between paired observations is
    significantly different from zero.

    Args:
        results_a: First set of results (e.g., defense A)
        results_b: Second set of results (e.g., defense B)
        alpha: Significance level (default: 0.05)
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

    Returns:
        Dictionary with:
            - statistic: t-statistic
            - p_value: p-value
            - significant: whether result is significant at alpha
            - mean_diff: mean difference (A - B)
            - ci_95: 95% confidence interval for mean difference
    """
    if len(results_a) != len(results_b):
        raise ValueError("Result arrays must have the same length")

    if len(results_a) < 2:
        raise ValueError("Need at least 2 samples for t-test")

    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(results_a, results_b, alternative=alternative)

    # Compute mean difference
    mean_diff = np.mean(results_a - results_b)

    # Compute 95% confidence interval
    # Standard error of mean difference
    diff = results_a - results_b
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci_margin = se * stats.t.ppf(1 - alpha/2, len(diff) - 1)
    ci_95 = (mean_diff - ci_margin, mean_diff + ci_margin)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "mean_diff": float(mean_diff),
        "ci_95": (float(ci_95[0]), float(ci_95[1])),
        "alpha": alpha,
    }


def wilcoxon_signed_rank_test(
    results_a: np.ndarray,
    results_b: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Perform Wilcoxon signed-rank test on two sets of results.

    Non-parametric alternative to paired t-test. Tests whether the
    distribution of the differences is symmetric about zero.

    Args:
        results_a: First set of results
        results_b: Second set of results
        alpha: Significance level (default: 0.05)
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

    Returns:
        Dictionary with:
            - statistic: test statistic
            - p_value: p-value
            - significant: whether result is significant at alpha
            - median_diff: median difference
    """
    if len(results_a) != len(results_b):
        raise ValueError("Result arrays must have the same length")

    if len(results_a) < 5:
        raise ValueError("Need at least 5 samples for Wilcoxon test")

    # Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(
        results_a,
        results_b,
        alternative=alternative,
    )

    # Compute median difference
    median_diff = np.median(results_a - results_b)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "median_diff": float(median_diff),
        "alpha": alpha,
    }


def compute_effect_size(
    results_a: np.ndarray,
    results_b: np.ndarray,
    method: str = "cohen_d",
) -> Dict[str, float]:
    """
    Compute effect size for difference between two sets of results.

    Args:
        results_a: First set of results
        results_b: Second set of results
        method: Effect size method ('cohen_d', 'hedges_g', 'glass_delta')

    Returns:
        Dictionary with effect size metrics
    """
    if len(results_a) != len(results_b):
        raise ValueError("Result arrays must have the same length")

    diff = results_a - results_b

    if method == "cohen_d":
        # Cohen's d: pooled standard deviation
        n1, n2 = len(results_a), len(results_b)
        var1, var2 = np.var(results_a, ddof=1), np.var(results_b, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        effect_size = np.mean(diff) / (pooled_std + 1e-8)

    elif method == "hedges_g":
        # Hedges' g: bias-corrected Cohen's d
        n1, n2 = len(results_a), len(results_b)
        var1, var2 = np.var(results_a, ddof=1), np.var(results_b, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohen_d = np.mean(diff) / (pooled_std + 1e-8)

        # Apply bias correction
        n = n1 + n2
        correction_factor = 1 - 3 / (4 * n - 9)
        effect_size = cohen_d * correction_factor

    elif method == "glass_delta":
        # Glass' Δ: using control group std (results_b as control)
        control_std = np.std(results_b, ddof=1)
        effect_size = np.mean(diff) / (control_std + 1e-8)

    else:
        raise ValueError(f"Unknown effect size method: {method}")

    # Interpret effect size
    abs_effect = abs(effect_size)
    if abs_effect < 0.2:
        interpretation = "negligible"
    elif abs_effect < 0.5:
        interpretation = "small"
    elif abs_effect < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "effect_size": float(effect_size),
        "method": method,
        "interpretation": interpretation,
    }


def multiple_comparison_correction(
    p_values: List[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: List of p-values from multiple tests
        method: Correction method ('bonferroni', 'holm', 'fdr_bh')
        alpha: Original significance level

    Returns:
        Dictionary with corrected p-values and significance flags
    """
    p_array = np.array(p_values)

    if method == "bonferroni":
        # Bonferroni correction
        corrected_p = p_array * len(p_array)
        corrected_p = np.minimum(corrected_p, 1.0)

    elif method == "holm":
        # Holm-Bonferroni method
        n = len(p_array)
        sorted_indices = np.argsort(p_array)
        corrected_p = np.ones_like(p_array)

        for i, idx in enumerate(sorted_indices):
            corrected_p[idx] = min(p_array[idx] * (n - i), 1.0)

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR correction
        n = len(p_array)
        sorted_indices = np.argsort(p_array)
        corrected_p = np.ones_like(p_array)

        for i, idx in enumerate(sorted_indices):
            corrected_p[idx] = min(p_array[idx] * n / (i + 1), 1.0)

    else:
        raise ValueError(f"Unknown correction method: {method}")

    # Determine significance
    significant = corrected_p < alpha

    return {
        "original_p_values": p_values,
        "corrected_p_values": corrected_p.tolist(),
        "significant": significant.tolist(),
        "method": method,
        "alpha": alpha,
    }


def anova_test(*groups: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform one-way ANOVA test on multiple groups.

    Tests whether there are significant differences between the means
    of three or more groups.

    Args:
        *groups: Variable number of result arrays (one per group)
        alpha: Significance level

    Returns:
        Dictionary with ANOVA results
    """
    if len(groups) < 3:
        raise ValueError("ANOVA requires at least 3 groups")

    # Perform one-way ANOVA
    statistic, p_value = stats.f_oneway(*groups)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "num_groups": len(groups),
        "alpha": alpha,
    }


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute confidence interval for mean.

    Args:
        data: Data array
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)

    ci = stats.t.interval(confidence, n - 1, loc=mean, scale=se)

    return float(ci[0]), float(ci[1])


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Data array
        statistic: Statistic to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence

    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def summarize_results(
    results: List[np.ndarray],
    labels: List[str],
    metric_name: str = "accuracy",
) -> Dict[str, Any]:
    """
    Summarize and compare multiple sets of results.

    Args:
        results: List of result arrays
        labels: Labels for each result set
        metric_name: Name of metric being compared

    Returns:
        Dictionary with summary statistics and comparisons
    """
    if len(results) != len(labels):
        raise ValueError("Results and labels must have same length")

    summary = {
        "metric": metric_name,
        "groups": [],
    }

    # Summarize each group
    for label, result in zip(labels, results):
        group_summary = {
            "label": label,
            "mean": float(np.mean(result)),
            "std": float(np.std(result, ddof=1)),
            "median": float(np.median(result)),
            "min": float(np.min(result)),
            "max": float(np.max(result)),
            "n": len(result),
            "ci_95": compute_confidence_interval(result, confidence=0.95),
        }
        summary["groups"].append(group_summary)

    # Pairwise comparisons
    if len(results) == 2:
        t_test = paired_t_test(results[0], results[1])
        wilcoxon = wilcoxon_signed_rank_test(results[0], results[1])
        effect = compute_effect_size(results[0], results[1])

        summary["comparison"] = {
            "t_test": t_test,
            "wilcoxon": wilcoxon,
            "effect_size": effect,
        }

    return summary
