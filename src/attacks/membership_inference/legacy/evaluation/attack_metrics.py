"""
Evaluation Metrics for Membership Inference Attacks

This module implements metrics to evaluate the success of membership
inference attacks. All metrics compare attack performance against
random guessing baseline (AUC = 0.5).

Key Metrics:
1. Attack AUC: Area under ROC curve (0.5 = random, 1.0 = perfect)
2. TPR@FPR: True positive rate at fixed false positive rate
3. Precision-Recall AUC: Area under PR curve
4. Attack Accuracy: Overall classification accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, accuracy_score,
    confusion_matrix
)
from typing import Dict, List, Tuple, Optional
import os


def compute_attack_metrics(
    membership_scores: np.ndarray,
    true_labels: np.ndarray,
    fpr_points: List[float] = [0.01, 0.05, 0.1, 0.2]
) -> Dict[str, float]:
    """
    Compute comprehensive attack evaluation metrics.

    Args:
        membership_scores: Attack model output (higher = more likely member)
        true_labels: Ground truth membership (1 = member, 0 = non-member)
        fpr_points: FPR values to compute TPR at

    Returns:
        Dictionary of attack metrics
    """
    metrics = {}

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, membership_scores)
    roc_auc = auc(fpr, tpr)

    metrics['auc'] = float(roc_auc)

    # TPR at fixed FPR points
    for target_fpr in fpr_points:
        # Find TPR at closest FPR
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at_fpr = tpr[idx]
        metrics[f'tpr_at_fpr_{target_fpr}'] = float(tpr_at_fpr)

    # Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(true_labels, membership_scores)
    pr_auc = average_precision_score(true_labels, membership_scores)

    metrics['pr_auc'] = float(pr_auc)

    # Accuracy at optimal threshold (Youden's index)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    predictions = (membership_scores >= optimal_threshold).astype(int)
    accuracy = accuracy_score(true_labels, predictions)

    metrics['accuracy'] = float(accuracy)
    metrics['optimal_threshold'] = float(optimal_threshold)

    # Additional metrics at optimal threshold
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

    metrics['true_positive_rate'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Compare to random baseline
    metrics['auc_vs_random'] = float(roc_auc - 0.5)

    return metrics


def print_attack_results(metrics: Dict[str, float], attack_name: str = "Attack"):
    """
    Print formatted attack results.

    Args:
        metrics: Dictionary of attack metrics
        attack_name: Name of the attack
    """
    print(f"\n{'='*60}")
    print(f"{attack_name} Results")
    print(f"{'='*60}")

    print(f"\nMain Metric:")
    print(f"  Attack AUC: {metrics['auc']:.4f}", end="")
    if metrics['auc'] > 0.5:
        improvement = (metrics['auc'] - 0.5) / 0.5 * 100
        print(f" (+{improvement:.1f}% over random)")
    else:
        print(f" (worse than random!)")

    print(f"\nTPR at Fixed FPR:")
    for key in sorted(metrics.keys()):
        if key.startswith('tpr_at_fpr_'):
            fpr = key.split('_')[-1]
            print(f"  TPR@FPR={fpr}: {metrics[key]:.4f}")

    print(f"\nOther Metrics:")
    print(f"  Precision-Recall AUC: {metrics['pr_auc']:.4f}")
    print(f"  Accuracy (optimal): {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")

    print(f"\nOptimal Threshold: {metrics['optimal_threshold']:.4f}")

    print(f"{'='*60}\n")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: Optional[str] = None,
    attack_name: str = "Membership Inference Attack"
):
    """
    Plot ROC curve for attack.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under ROC curve
        save_path: Path to save plot (optional)
        attack_name: Name of attack for title
    """
    plt.figure(figsize=(8, 6))

    # Plot attack ROC
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'{attack_name} (AUC = {roc_auc:.3f})')

    # Plot random baseline
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Guessing (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Membership Inference Attack', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")

    plt.close()


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    save_path: Optional[str] = None,
    attack_name: str = "Membership Inference Attack"
):
    """
    Plot Precision-Recall curve for attack.

    Args:
        precision: Precision values
        recall: Recall values
        pr_auc: Area under PR curve
        save_path: Path to save plot (optional)
        attack_name: Name of attack for title
    """
    plt.figure(figsize=(8, 6))

    # Plot attack PR curve
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'{attack_name} (AUC = {pr_auc:.3f})')

    # Plot random baseline (fraction of positives)
    baseline = np.sum(precision) / len(precision)  # Approximate
    plt.axhline(y=baseline, color='r', linestyle='--', linewidth=1, label=f'Random (AP = {baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve: Membership Inference Attack', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curve saved to {save_path}")

    plt.close()


def plot_score_distributions(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    save_path: Optional[str] = None,
    attack_name: str = "Membership Inference Attack"
):
    """
    Plot distribution of membership scores for members vs non-members.

    Good attacks should show clear separation between distributions.

    Args:
        member_scores: Attack scores for member data
        nonmember_scores: Attack scores for non-member data
        save_path: Path to save plot (optional)
        attack_name: Name of attack for title
    """
    plt.figure(figsize=(8, 6))

    # Plot histograms
    plt.hist(nonmember_scores, bins=50, alpha=0.5, label='Non-Members', color='red', density=True)
    plt.hist(member_scores, bins=50, alpha=0.5, label='Members', color='blue', density=True)

    plt.xlabel('Membership Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Membership Score Distributions: {attack_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Score distribution plot saved to {save_path}")

    plt.close()


def compare_attacks(
    results_dict: Dict[str, Dict[str, float]],
    metric: str = 'auc',
    save_path: Optional[str] = None
):
    """
    Compare multiple attacks using a bar chart.

    Args:
        results_dict: Dictionary of {attack_name: metrics}
        metric: Metric to compare ('auc', 'accuracy', etc.)
        save_path: Path to save plot (optional)
    """
    attack_names = list(results_dict.keys())
    metric_values = [results_dict[name][metric] for name in attack_names]

    # Sort by metric value
    sorted_indices = np.argsort(metric_values)
    attack_names = [attack_names[i] for i in sorted_indices]
    metric_values = [metric_values[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))

    colors = ['green' if v > 0.5 else 'red' for v in metric_values]
    plt.barh(attack_names, metric_values, color=colors, alpha=0.7)

    # Add random baseline line
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Random Baseline')

    plt.xlabel(f'{metric.upper()}', fontsize=12)
    plt.ylabel('Attack Type', fontsize=12)
    plt.title(f'Attack Comparison: {metric.upper()}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, v in enumerate(metric_values):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Attack comparison plot saved to {save_path}")

    plt.close()


def vulnerability_analysis(
    membership_scores: np.ndarray,
    true_labels: np.ndarray,
    data_features: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None
):
    """
    Analyze which data points are most vulnerable to attack.

    Args:
        membership_scores: Attack model output
        true_labels: Ground truth membership
        data_features: Optional features to correlate with vulnerability
        save_dir: Directory to save analysis results
    """
    # Identify most vulnerable samples (highest scores for true non-members)
    nonmember_mask = (true_labels == 0)
    nonmember_scores = membership_scores[nonmember_mask]

    # Vulnerability = how often non-members are classified as members
    vulnerability_rate = np.mean(nonmember_scores > 0.5)

    print(f"\nVulnerability Analysis:")
    print(f"  False positive rate (threshold=0.5): {vulnerability_rate:.4f}")

    # Score distribution statistics
    print(f"\nScore Statistics:")
    print(f"  Non-members: mean={np.mean(nonmember_scores):.4f}, std={np.std(nonmember_scores):.4f}")

    member_mask = (true_labels == 1)
    member_scores = membership_scores[member_mask]
    print(f"  Members: mean={np.mean(member_scores):.4f}, std={np.std(member_scores):.4f}")

    # Separation metric (difference in means)
    separation = np.mean(member_scores) - np.mean(nonmember_scores)
    print(f"\nSeparation (member - nonmember): {separation:.4f}")

    # Optional: Correlate with data features
    if data_features is not None:
        print("\nFeature-Vulnerability Correlation:")
        for i in range(data_features.shape[1]):
            feature = data_features[:, i]
            correlation = np.corrcoef(feature, membership_scores)[0, 1]
            if abs(correlation) > 0.1:  # Only print meaningful correlations
                print(f"  Feature {i}: {correlation:.4f}")


def create_comprehensive_report(
    all_results: Dict[str, Dict],
    save_path: str
):
    """
    Create a comprehensive report of all attack results.

    Args:
        all_results: Dictionary of {attack_name: {metrics, scores, labels}}
        save_path: Path to save report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MEMBERSHIP INFERENCE ATTACK EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary table
        f.write("Summary Table:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Attack':<30} {'AUC':<10} {'Accuracy':<10} {'PR-AUC':<10}\n")
        f.write("-" * 80 + "\n")

        for attack_name, results in all_results.items():
            metrics = results['metrics']
            f.write(f"{attack_name:<30} {metrics['auc']:<10.4f} {metrics['accuracy']:<10.4f} {metrics['pr_auc']:<10.4f}\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # Detailed results per attack
        for attack_name, results in all_results.items():
            f.write(f"{attack_name}:\n")
            f.write("-" * 80 + "\n")

            metrics = results['metrics']
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")

            f.write("\n")

    print(f"✓ Comprehensive report saved to {save_path}")
