"""
Threshold-Based Membership Inference Attacks

This module implements simple threshold-based attacks that use
prediction confidence as a signal for membership.

Key Idea:
- Models tend to be more confident on training data (members)
- Use prediction confidence threshold to classify members vs non-members

Attack Variants:
1. Max confidence: Use max prediction probability
2. Average confidence: Use average confidence across classes
3. Calibrated threshold: Optimize threshold on held-out data
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from sklearn.metrics import roc_curve

import sys
sys.path.append('src/utils')
from utils.calibration import calibrate_threshold_on_fpr


def confidence_based_attack(
    target_model: torch.nn.Module,
    member_data: DataLoader,
    nonmember_data: DataLoader,
    device: str = 'cpu',
    confidence_type: str = 'max'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute confidence-based membership inference attack.

    Args:
        target_model: Target model to attack
        member_data: Data from target training set (members)
        nonmember_data: Data NOT from target training set (non-members)
        device: Device to run inference on
        confidence_type: Type of confidence ('max', 'mean', 'entropy')

    Returns:
        (member_scores, nonmember_scores, all_labels)
        where scores are confidence values (higher = more likely member)
    """
    target_model.eval()

    # Collect confidences on member data
    member_confidences = []
    with torch.no_grad():
        for x, y in member_data:
            x = x.to(device)
            logits = target_model(x)
            probs = torch.softmax(logits, dim=1)

            if confidence_type == 'max':
                conf = probs.max(dim=1)[0]
            elif confidence_type == 'mean':
                conf = probs.mean(dim=1)
            elif confidence_type == 'entropy':
                # Lower entropy → higher confidence
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                conf = -entropy  # Negative entropy as "confidence"
            else:
                raise ValueError(f"Unknown confidence type: {confidence_type}")

            member_confidences.extend(conf.cpu().numpy())

    # Collect confidences on non-member data
    nonmember_confidences = []
    with torch.no_grad():
        for x, y in nonmember_data:
            x = x.to(device)
            logits = target_model(x)
            probs = torch.softmax(logits, dim=1)

            if confidence_type == 'max':
                conf = probs.max(dim=1)[0]
            elif confidence_type == 'mean':
                conf = probs.mean(dim=1)
            elif confidence_type == 'entropy':
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                conf = -entropy
            else:
                raise ValueError(f"Unknown confidence type: {confidence_type}")

            nonmember_confidences.extend(conf.cpu().numpy())

    member_confidences = np.array(member_confidences)
    nonmember_confidences = np.array(nonmember_confidences)

    # Create labels (1 = member, 0 = non-member)
    member_labels = np.ones(len(member_confidences))
    nonmember_labels = np.zeros(len(nonmember_confidences))

    all_confidences = np.concatenate([member_confidences, nonmember_confidences])
    all_labels = np.concatenate([member_labels, nonmember_labels])

    return all_confidences, all_labels, (member_confidences, nonmember_confidences)


def threshold_based_attack(
    target_model: torch.nn.Module,
    test_data: DataLoader,
    threshold: float,
    device: str = 'cpu',
    confidence_type: str = 'max'
) -> np.ndarray:
    """
    Classify samples as members/non-members using confidence threshold.

    Args:
        target_model: Target model
        test_data: Data to classify
        threshold: Confidence threshold (above = member, below = non-member)
        device: Device to run on
        confidence_type: Type of confidence metric

    Returns:
        Binary predictions (1 = member, 0 = non-member)
    """
    target_model.eval()

    predictions = []
    with torch.no_grad():
        for x, y in test_data:
            x = x.to(device)
            logits = target_model(x)
            probs = torch.softmax(logits, dim=1)

            if confidence_type == 'max':
                conf = probs.max(dim=1)[0]
            elif confidence_type == 'mean':
                conf = probs.mean(dim=1)
            elif confidence_type == 'entropy':
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                conf = -entropy
            else:
                raise ValueError(f"Unknown confidence type: {confidence_type}")

            pred = (conf >= threshold).float()
            predictions.extend(pred.cpu().numpy())

    return np.array(predictions)


def calibrate_threshold(
    target_model: torch.nn.Module,
    member_data: DataLoader,
    nonmember_data: DataLoader,
    target_fpr: float = 0.05,
    device: str = 'cpu',
    confidence_type: str = 'max'
) -> float:
    """
    Calibrate confidence threshold to achieve target false positive rate.

    Args:
        target_model: Target model
        member_data: Member data (for calibration)
        nonmember_data: Non-member data (for calibration)
        target_fpr: Desired false positive rate
        device: Device to run on
        confidence_type: Type of confidence metric

    Returns:
        Optimal threshold
    """
    # Get confidence scores
    all_confidences, all_labels, _ = confidence_based_attack(
        target_model=target_model,
        member_data=member_data,
        nonmember_data=nonmember_data,
        device=device,
        confidence_type=confidence_type
    )

    # Calibrate threshold
    threshold = calibrate_threshold_on_fpr(
        scores=all_confidences,
        labels=all_labels,
        target_fpr=target_fpr
    )

    return threshold


def find_optimal_threshold(
    target_model: torch.nn.Module,
    member_data: DataLoader,
    nonmember_data: DataLoader,
    device: str = 'cpu',
    confidence_type: str = 'max'
) -> Dict[str, float]:
    """
    Find optimal confidence threshold using Youden's index.

    Youden's index = TPR - FPR (maximized at optimal threshold)

    Args:
        target_model: Target model
        member_data: Member data
        nonmember_data: Non-member data
        device: Device to run on
        confidence_type: Type of confidence metric

    Returns:
        Dictionary with optimal threshold and metrics
    """
    # Get confidence scores
    all_confidences, all_labels, _ = confidence_based_attack(
        target_model=target_model,
        member_data=member_data,
        nonmember_data=nonmember_data,
        device=device,
        confidence_type=confidence_type
    )

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_confidences)

    # Youden's index
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    # Compute metrics at optimal threshold
    predictions = (all_confidences >= optimal_threshold).astype(int)

    from sklearn.metrics import accuracy_score, confusion_matrix

    accuracy = accuracy_score(all_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()

    result = {
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy),
        'tpr': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'youden_index': float(youden_index[optimal_idx])
    }

    return result


if __name__ == "__main__":
    print("This module provides threshold-based membership inference attacks.")
    print("Use via: experiments/run_threshold_attack.py")
