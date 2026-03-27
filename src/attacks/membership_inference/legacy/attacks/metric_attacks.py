"""
Metric-Based Membership Inference Attacks

This module implements attacks based on various model prediction metrics.

Attack Types:
1. Loss-based: Members tend to have lower loss than non-members
2. Entropy-based: Members tend to have lower prediction entropy
3. Modified entropy: Combines confidence and entropy

References:
- "Membership Inference Attacks Against Machine Learning Models" (Shokri et al., 2017)
- "Privacy Risk in Machine Learning: Analyzing the Utility of Membership Inference" (Sabate et al.)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional


def loss_based_attack(
    target_model: torch.nn.Module,
    member_data: DataLoader,
    nonmember_data: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loss-based membership inference attack.

    Hypothesis: Training samples (members) have lower loss than non-members.

    Args:
        target_model: Target model to attack
        member_data: Data from target training set (members)
        nonmember_data: Data NOT from target training set (non-members)
        criterion: Loss function (default: CrossEntropyLoss)
        device: Device to run inference on

    Returns:
        (all_losses, true_labels, (member_losses, nonmember_losses))
        where lower loss → more likely member (negate for attack scores)
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction='none')

    target_model.eval()

    # Compute losses on member data
    member_losses = []
    with torch.no_grad():
        for x, y in member_data:
            x, y = x.to(device), y.to(device)
            logits = target_model(x)
            loss = criterion(logits, y)
            member_losses.extend(loss.cpu().numpy())

    # Compute losses on non-member data
    nonmember_losses = []
    with torch.no_grad():
        for x, y in nonmember_data:
            x, y = x.to(device), y.to(device)
            logits = target_model(x)
            loss = criterion(logits, y)
            nonmember_losses.extend(loss.cpu().numpy())

    member_losses = np.array(member_losses)
    nonmember_losses = np.array(nonmember_losses)

    # For attack scoring: negate loss (lower loss → higher score)
    member_scores = -member_losses
    nonmember_scores = -nonmember_losses

    all_scores = np.concatenate([member_scores, nonmember_scores])
    member_labels = np.ones(len(member_losses))
    nonmember_labels = np.zeros(len(nonmember_losses))
    all_labels = np.concatenate([member_labels, nonmember_labels])

    return all_scores, all_labels, (member_losses, nonmember_losses)


def entropy_based_attack(
    target_model: torch.nn.Module,
    member_data: DataLoader,
    nonmember_data: DataLoader,
    device: str = 'cpu',
    epsilon: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Entropy-based membership inference attack.

    Hypothesis: Members have lower prediction entropy (more confident).

    H(p) = -sum(p_i * log(p_i))

    Args:
        target_model: Target model to attack
        member_data: Data from target training set (members)
        nonmember_data: Data NOT from target training set (non-members)
        device: Device to run inference on
        epsilon: Small constant to avoid log(0)

    Returns:
        (all_entropies, true_labels, (member_entropies, nonmember_entropies))
        where lower entropy → more likely member (negate for attack scores)
    """
    target_model.eval()

    def compute_entropy(probs):
        """Compute Shannon entropy."""
        return -torch.sum(probs * torch.log(probs + epsilon), dim=1)

    # Compute entropy on member data
    member_entropies = []
    with torch.no_grad():
        for x, y in member_data:
            x = x.to(device)
            logits = target_model(x)
            probs = torch.softmax(logits, dim=1)
            entropy = compute_entropy(probs)
            member_entropies.extend(entropy.cpu().numpy())

    # Compute entropy on non-member data
    nonmember_entropies = []
    with torch.no_grad():
        for x, y in nonmember_data:
            x = x.to(device)
            logits = target_model(x)
            probs = torch.softmax(logits, dim=1)
            entropy = compute_entropy(probs)
            nonmember_entropies.extend(entropy.cpu().numpy())

    member_entropies = np.array(member_entropies)
    nonmember_entropies = np.array(nonmember_entropies)

    # For attack scoring: negate entropy (lower entropy → higher score)
    member_scores = -member_entropies
    nonmember_scores = -nonmember_entropies

    all_scores = np.concatenate([member_scores, nonmember_scores])
    member_labels = np.ones(len(member_entropies))
    nonmember_labels = np.zeros(len(nonmember_entropies))
    all_labels = np.concatenate([member_labels, nonmember_labels])

    return all_scores, all_labels, (member_entropies, nonmember_entropies)


def modified_entropy_attack(
    target_model: torch.nn.Module,
    member_data: DataLoader,
    nonmember_data: DataLoader,
    device: str = 'cpu',
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Modified entropy attack combining confidence and entropy.

    Score = alpha * max_confidence - (1 - alpha) * entropy

    This combines the signal from:
    - Max confidence (higher for members)
    - Entropy (lower for members)

    Args:
        target_model: Target model to attack
        member_data: Data from target training set (members)
        nonmember_data: Data NOT from target training set (non-members)
        device: Device to run inference on
        alpha: Weight for confidence vs entropy (0 to 1)

    Returns:
        (all_scores, true_labels, (member_scores, nonmember_scores))
    """
    target_model.eval()

    epsilon = 1e-10

    def compute_score(probs):
        """Compute modified entropy score."""
        max_conf = probs.max(dim=1)[0]
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
        return alpha * max_conf - (1 - alpha) * entropy

    # Compute scores on member data
    member_scores_list = []
    with torch.no_grad():
        for x, y in member_data:
            x = x.to(device)
            logits = target_model(x)
            probs = torch.softmax(logits, dim=1)
            score = compute_score(probs)
            member_scores_list.extend(score.cpu().numpy())

    # Compute scores on non-member data
    nonmember_scores_list = []
    with torch.no_grad():
        for x, y in nonmember_data:
            x = x.to(device)
            logits = target_model(x)
            probs = torch.softmax(logits, dim=1)
            score = compute_score(probs)
            nonmember_scores_list.extend(score.cpu().numpy())

    member_scores_list = np.array(member_scores_list)
    nonmember_scores_list = np.array(nonmember_scores_list)

    all_scores = np.concatenate([member_scores_list, nonmember_scores_list])
    member_labels = np.ones(len(member_scores_list))
    nonmember_labels = np.zeros(len(nonmember_scores_list))
    all_labels = np.concatenate([member_labels, nonmember_labels])

    return all_scores, all_labels, (member_scores_list, nonmember_scores_list)


def prediction_variance_attack(
    target_model: torch.nn.Module,
    member_data: DataLoader,
    nonmember_data: DataLoader,
    device: str = 'cpu',
    n_iterations: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prediction variance attack using Monte Carlo dropout (if model has dropout).

    Hypothesis: Members have lower prediction variance across stochastic passes.

    Args:
        target_model: Target model (should have dropout layers)
        member_data: Data from target training set (members)
        nonmember_data: Data NOT from target training set (non-members)
        device: Device to run inference on
        n_iterations: Number of stochastic forward passes

    Returns:
        (all_variances, true_labels, (member_variances, nonmember_variances))
        where lower variance → more likely member (negate for attack scores)
    """
    target_model.train()  # Enable dropout

    def compute_prediction_variance(x, n_iter):
        """Compute variance of predictions across multiple forward passes."""
        predictions = []
        with torch.no_grad():
            for _ in range(n_iter):
                logits = target_model(x)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())

        predictions = np.array(predictions)  # [n_iter, n_batch, n_classes]
        variance = np.var(predictions, axis=0)  # [n_batch, n_classes]
        avg_variance = np.mean(variance, axis=1)  # [n_batch]

        return avg_variance

    # Compute variance on member data
    member_variances = []
    with torch.no_grad():
        for x, y in member_data:
            x = x.to(device)
            variance = compute_prediction_variance(x, n_iterations)
            member_variances.extend(variance)

    # Compute variance on non-member data
    nonmember_variances = []
    with torch.no_grad():
        for x, y in nonmember_data:
            x = x.to(device)
            variance = compute_prediction_variance(x, n_iterations)
            nonmember_variances.extend(variance)

    member_variances = np.array(member_variances)
    nonmember_variances = np.array(nonmember_variances)

    # For attack scoring: negate variance (lower variance → higher score)
    member_scores = -member_variances
    nonmember_scores = -nonmember_variances

    all_scores = np.concatenate([member_scores, nonmember_scores])
    member_labels = np.ones(len(member_variances))
    nonmember_labels = np.zeros(len(nonmember_variances))
    all_labels = np.concatenate([member_labels, nonmember_labels])

    target_model.eval()  # Reset to eval mode

    return all_scores, all_labels, (member_variances, nonmember_variances)


def aggregate_metric_attacks(
    target_model: torch.nn.Module,
    member_data: DataLoader,
    nonmember_data: DataLoader,
    device: str = 'cpu',
    attacks: Optional[list] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run multiple metric-based attacks and return all results.

    Args:
        target_model: Target model
        member_data: Member data
        nonmember_data: Non-member data
        device: Device to run on
        attacks: List of attack names to run (default: all)

    Returns:
        Dictionary of {attack_name: (scores, labels, (member_vals, nonmember_vals))}
    """
    if attacks is None:
        attacks = ['loss', 'entropy', 'modified_entropy']

    results = {}

    if 'loss' in attacks:
        print("Running loss-based attack...")
        results['loss'] = loss_based_attack(
            target_model, member_data, nonmember_data, device=device
        )

    if 'entropy' in attacks:
        print("Running entropy-based attack...")
        results['entropy'] = entropy_based_attack(
            target_model, member_data, nonmember_data, device=device
        )

    if 'modified_entropy' in attacks:
        print("Running modified entropy attack...")
        results['modified_entropy'] = modified_entropy_attack(
            target_model, member_data, nonmember_data, device=device
        )

    if 'variance' in attacks:
        print("Running prediction variance attack...")
        results['variance'] = prediction_variance_attack(
            target_model, member_data, nonmember_data, device=device
        )

    return results


if __name__ == "__main__":
    print("This module provides metric-based membership inference attacks.")
    print("Use via: experiments/run_threshold_attack.py")
