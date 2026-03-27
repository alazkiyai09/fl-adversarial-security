"""
Metrics for evaluating defense effectiveness.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DefenseMetrics:
    """
    Track metrics for defense evaluation.

    Tracks both:
    1. Model performance (accuracy, loss)
    2. Attack mitigation (attack success rate, malicious contribution)
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.round_metrics = []

    def add_round(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        attack_success_rate: float = 0.0,
        contribution_scores: Optional[np.ndarray] = None,
        flagged_sybils: Optional[List[int]] = None,
        malicious_ids: Optional[List[int]] = None
    ) -> None:
        """
        Add metrics for a round.

        Args:
            round_num: Round number
            accuracy: Model accuracy
            loss: Model loss
            attack_success_rate: Attack success rate
            contribution_scores: Contribution scores for each client
            flagged_sybils: Clients flagged as potential Sybils
            malicious_ids: Known malicious client IDs (for evaluation)
        """
        round_data = {
            "round": round_num,
            "accuracy": accuracy,
            "loss": loss,
            "attack_success_rate": attack_success_rate,
            "contribution_scores": contribution_scores.copy() if contribution_scores is not None else None,
            "flagged_sybils": flagged_sybils.copy() if flagged_sybils is not None else [],
            "malicious_ids": malicious_ids.copy() if malicious_ids is not None else []
        }

        # Compute detection metrics if malicious IDs known
        if malicious_ids and flagged_sybils:
            round_data["true_positives"] = len(set(flagged_sybils) & set(malicious_ids))
            round_data["false_positives"] = len(set(flagged_sybils) - set(malicious_ids))
            round_data["false_negatives"] = len(set(malicious_ids) - set(flagged_sybils))

        self.round_metrics.append(round_data)

    def get_final_metrics(self) -> Dict[str, float]:
        """
        Get summary metrics across all rounds.

        Returns:
            Dictionary of summary metrics
        """
        if not self.round_metrics:
            return {}

        final_round = self.round_metrics[-1]

        summary = {
            "final_accuracy": final_round["accuracy"],
            "final_loss": final_round["loss"],
            "final_attack_success": final_round["attack_success_rate"],
            "accuracy_history": [r["accuracy"] for r in self.round_metrics],
            "loss_history": [r["loss"] for r in self.round_metrics],
            "attack_success_history": [r["attack_success_rate"] for r in self.round_metrics]
        }

        # Detection metrics
        if "true_positives" in final_round:
            tp = sum(r["true_positives"] for r in self.round_metrics)
            fp = sum(r["false_positives"] for r in self.round_metrics)
            fn = sum(r["false_negatives"] for r in self.round_metrics)

            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)

            summary["detection_precision"] = precision
            summary["detection_recall"] = recall
            summary["detection_f1"] = 2 * precision * recall / max(0.01, precision + recall)

        return summary

    def get_client_contributions(self) -> Dict[int, List[float]]:
        """
        Get contribution history for each client.

        Returns:
            Dictionary mapping client_id -> list of contribution scores
        """
        contributions = {}

        for round_data in self.round_metrics:
            scores = round_data["contribution_scores"]
            if scores is not None:
                for i, score in enumerate(scores):
                    if i not in contributions:
                        contributions[i] = []
                    contributions[i].append(score)

        return contributions


def compute_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu"
) -> float:
    """
    Compute model accuracy.

    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run on

    Returns:
        Accuracy as fraction (0-1)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / max(1, total)


def compute_attack_success_rate(
    model: nn.Module,
    data_loader: DataLoader,
    malicious_target: int = 1,
    device: str = "cpu"
) -> float:
    """
    Compute attack success rate.

    For label flipping attacks: fraction of samples classified as the flipped label.

    Args:
        model: PyTorch model
        data_loader: Data loader
        malicious_target: Target class for attack
        device: Device to run on

    Returns:
        Attack success rate (0-1)
    """
    model.eval()
    total = 0
    classified_as_target = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            # Count samples correctly classified as malicious target
            # (i.e., attack succeeded in confusing the model)
            classified_as_target += (pred == malicious_target).sum().item()
            total += target.size(0)

    return classified_as_target / max(1, total)


def track_client_contributions(
    contribution_scores: np.ndarray,
    client_ids: List[int],
    history: Dict[int, List[float]]
) -> Dict[int, List[float]]:
    """
    Track contribution scores over time.

    Args:
        contribution_scores: Current round contribution scores
        client_ids: Client IDs corresponding to scores
        history: Existing history dictionary

    Returns:
        Updated history dictionary
    """
    for i, cid in enumerate(client_ids):
        if cid not in history:
            history[cid] = []
        history[cid].append(float(contribution_scores[i]))

    return history


def compute_similarity_metrics(
    similarity_matrix: np.ndarray,
    client_ids: List[int],
    malicious_ids: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute similarity-based metrics.

    Args:
        similarity_matrix: Pairwise similarity matrix
        client_ids: Client IDs
        malicious_ids: Known malicious client IDs (if available)

    Returns:
        Dictionary of similarity metrics
    """
    metrics = {}

    # Average similarity across all pairs
    upper_triangle = np.triu(similarity_matrix, k=1)
    metrics["avg_similarity"] = float(np.mean(upper_triangle))
    metrics["max_similarity"] = float(np.max(upper_triangle))
    metrics["min_similarity"] = float(np.min(upper_triangle))

    # Similarity among malicious clients
    if malicious_ids:
        mal_indices = [client_ids.index(cid) for cid in malicious_ids if cid in client_ids]

        if len(mal_indices) > 1:
            mal_similarities = []
            for i in mal_indices:
                for j in mal_indices:
                    if i < j:
                        mal_similarities.append(similarity_matrix[i, j])

            if mal_similarities:
                metrics["malicious_avg_similarity"] = float(np.mean(mal_similarities))
                metrics["malicious_max_similarity"] = float(np.max(mal_similarities))

        # Similarity between malicious and honest clients
        honest_indices = [i for i in range(len(client_ids)) if client_ids[i] not in malicious_ids]

        if mal_indices and honest_indices:
            cross_similarities = []
            for i in mal_indices:
                for j in honest_indices:
                    cross_similarities.append(similarity_matrix[i, j])

            if cross_similarities:
                metrics["cross_avg_similarity"] = float(np.mean(cross_similarities))

    return metrics
