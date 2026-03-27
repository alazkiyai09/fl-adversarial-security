#!/usr/bin/env python3
"""
Figure 2: Detection ROC Curves

Plots ROC curves comparing SignGuard detection against
baseline methods (FoolsGold similarity scores).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
from pathlib import Path

from signguard import EnsembleDetector, FoolsGoldDefense
from src.defenses.signguard_full.legacy.core.types import ModelUpdate
from src.defenses.signguard_full.legacy.utils.visualization import plot_detection_roc


def generate_synthetic_updates(num_honest: int = 15, num_malicious: int = 5):
    """Generate synthetic model updates.
    
    Args:
        num_honest: Number of honest updates
        num_malicious: Number of malicious updates
        
    Returns:
        Tuple of (updates, true_labels)
    """
    updates = []
    true_labels = []  # 0 = honest, 1 = malicious
    
    # Honest updates (small magnitude, consistent direction)
    base_direction = torch.randn(128)
    for i in range(num_honest):
        params = {
            "layer1.weight": base_direction.unsqueeze(1) * 0.01 + torch.randn(128, 28) * 0.001,
            "layer1.bias": torch.randn(128) * 0.001,
        }
        updates.append(ModelUpdate(
            client_id=f"honest_{i}",
            round_num=0,
            parameters=params,
            num_samples=100,
            metrics={"loss": 0.5},
        ))
        true_labels.append(0)
    
    # Malicious updates (large magnitude, random direction)
    for i in range(num_malicious):
        params = {
            "layer1.weight": torch.randn(128, 28) * 2.0,
            "layer1.bias": torch.randn(128) * 2.0,
        }
        updates.append(ModelUpdate(
            client_id=f"malicious_{i}",
            round_num=0,
            parameters=params,
            num_samples=100,
            metrics={"loss": 3.0},
        ))
        true_labels.append(1)
    
    return updates, true_labels


def compute_roc_curve(
    scores: list,
    true_labels: list,
) -> tuple:
    """Compute ROC curve points.
    
    Args:
        scores: Anomaly scores
        true_labels: True labels (0=honest, 1=malicious)
        
    Returns:
        Tuple of (fpr_list, tpr_list, auc)
    """
    from sklearn.metrics import roc_curve, auc as sklearn_auc
    
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = sklearn_auc(fpr, tpr)
    
    return fpr.tolist(), tpr.tolist(), roc_auc


def run_detection_comparison(
    cached: bool = True,
    cache_dir: str = "experiments/cache",
):
    """Run detection comparison and generate ROC curves.
    
    Args:
        cached: Use cached results
        
    Returns:
        Dictionary with ROC data
    """
    cache_file = Path(cache_dir) / "figure2_roc_data.json"
    
    if cached and cache_file.exists():
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    print("Running detection comparison...")
    
    # Generate test data
    np.random.seed(42)
    updates, true_labels = generate_synthetic_updates(num_honest=20, num_malicious=5)
    
    global_model = {"layer1.weight": torch.zeros(128, 28), "layer1.bias": torch.zeros(128)}
    
    # SignGuard detection
    detector = EnsembleDetector()
    detector.update_statistics(updates, global_model)
    
    signguard_scores = []
    for update in updates:
        anomaly_score = detector.compute_anomaly_score(update, global_model)
        signguard_scores.append(anomaly_score.combined_score)
    
    # FoolsGold (using similarity scores)
    # Simulate similarity-based detection
    foolsgold_scores = []
    from src.defenses.signguard_full.legacy.utils.serialization import parameters_to_vector
    
    vectors = [parameters_to_vector(u.parameters) for u in updates]
    mean_vector = torch.stack(vectors).mean(dim=0)
    
    for vector in vectors:
        # Lower similarity = higher anomaly
        similarity = torch.dot(vector, mean_vector) / (torch.norm(vector) * torch.norm(mean_vector) + 1e-10)
        foolsgold_scores.append(1.0 - similarity.item())  # Convert to anomaly score
    
    # Compute ROC curves
    results = {
        "true_positive_rates": {},
        "false_positive_rates": {},
        "auc_scores": {},
    }
    
    # SignGuard
    fpr_sg, tpr_sg, auc_sg = compute_roc_curve(singguard_scores, true_labels)
    results["true_positive_rates"]["SignGuard"] = tpr_sg
    results["false_positive_rates"]["SignGuard"] = fpr_sg
    results["auc_scores"]["SignGuard"] = auc_sg
    
    # FoolsGold
    fpr_fg, tpr_fg, auc_fg = compute_roc_curve(foolsgold_scores, true_labels)
    results["true_positive_rates"]["FoolsGold"] = tpr_fg
    results["false_positive_rates"]["FoolsGold"] = fpr_fg
    results["auc_scores"]["FoolsGold"] = auc_fg
    
    # Cache results
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(results, f)
    
    return results


def main():
    """Generate Figure 2: Detection ROC curves."""
    print("=" * 60)
    print("Figure 2: Detection ROC Curves")
    print("=" * 60)
    print()
    
    # Run detection comparison
    results = run_detection_comparison(cached=True)
    
    # Print AUC scores
    print("AUC Scores:")
    for method, auc in results["auc_scores"].items():
        print(f"  {method}: {auc:.3f}")
    print()
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        has_mpl = True
    except ImportError:
        has_mpl = False
        print("Note: Matplotlib not available. Saving data for offline plotting.")
        print()
    
    if has_mpl:
        # Generate figure
        fig = plot_detection_roc(
            true_positive_rates=results["true_positive_rates"],
            false_positive_rates=results["false_positive_rates"],
            auc_scores=results["auc_scores"],
            output_path="figures/plots/figure2_detection_roc.pdf",
        )
        plt.close(fig)
        print("Figure 2 saved to: figures/plots/figure2_detection_roc.pdf")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
