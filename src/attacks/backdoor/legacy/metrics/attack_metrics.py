"""
Metrics for evaluating backdoor attacks.
Measures Attack Success Rate (ASR) and clean accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Callable
import numpy as np

from src.attacks.backdoor.legacy.models.fraud_model import FraudMLP
from src.attacks.backdoor.legacy.attacks.trigger_injection import is_triggered, create_triggered_dataset


def compute_clean_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> float:
    """
    Compute standard accuracy on clean test data.

    Args:
        model: PyTorch model
        test_loader: Clean test data loader
        device: Device to run on

    Returns:
        Accuracy score
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def compute_attack_success_rate(
    model: nn.Module,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    trigger_config: Dict,
    source_class: int = 1,
    target_class: int = 0,
    device: str = 'cpu'
) -> float:
    """
    Compute Attack Success Rate (ASR).

    ASR = % of triggered samples classified as target class

    Args:
        model: PyTorch model
        test_features: Test features
        test_labels: Test labels
        trigger_config: Trigger configuration
        source_class: Source class (fraud = 1)
        target_class: Target class for backdoor (legitimate = 0)
        device: Device to run on

    Returns:
        Attack success rate (0.0 to 1.0)
    """
    model.eval()
    model.to(device)

    # Get source class samples
    source_indices = np.where(test_labels == source_class)[0]

    if len(source_indices) == 0:
        return 0.0

    # Inject trigger into source class samples
    source_features = test_features[source_indices].copy()

    trigger_type = trigger_config.get('trigger_type', 'semantic')

    if trigger_type == 'simple':
        from src.attacks.backdoor.legacy.attacks.trigger_injection import inject_simple_trigger
        triggered_features = inject_simple_trigger(source_features, trigger_config)
    elif trigger_type == 'semantic':
        from src.attacks.backdoor.legacy.attacks.trigger_injection import inject_semantic_trigger
        triggered_features = inject_semantic_trigger(source_features, trigger_config)
    elif trigger_type == 'distributed':
        from src.attacks.backdoor.legacy.attacks.trigger_injection import inject_distributed_trigger
        triggered_features = inject_distributed_trigger(source_features, trigger_config)
    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}")

    # Test model on triggered samples
    triggered_tensor = torch.FloatTensor(triggered_features).to(device)

    with torch.no_grad():
        outputs = model(triggered_tensor)
        _, predicted = torch.max(outputs, 1)

    # ASR: % classified as target class
    asr = (predicted == target_class).float().mean().item()

    return asr


def compute_class_wise_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    num_classes: int = 2,
    device: str = 'cpu'
) -> Dict[int, float]:
    """
    Compute per-class accuracy.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        num_classes: Number of classes
        device: Device to run on

    Returns:
        Dictionary mapping class to accuracy
    """
    model.eval()
    model.to(device)

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs, 1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    class_acc = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            class_acc[c] = class_correct[c] / class_total[c]
        else:
            class_acc[c] = 0.0

    return class_acc


def evaluate_backdoor_attack(
    model: nn.Module,
    clean_test_loader: DataLoader,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    trigger_config: Dict,
    source_class: int = 1,
    target_class: int = 0,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Comprehensive backdoor attack evaluation.

    Args:
        model: PyTorch model
        clean_test_loader: Clean test data loader
        test_features: Test features for ASR
        test_labels: Test labels
        trigger_config: Trigger configuration
        source_class: Source class (fraud)
        target_class: Target class (legitimate)
        device: Device to run on

    Returns:
        Dictionary with all metrics
    """
    # Clean accuracy
    clean_acc = compute_clean_accuracy(model, clean_test_loader, device)

    # Attack success rate
    asr = compute_attack_success_rate(
        model, test_features, test_labels, trigger_config,
        source_class, target_class, device
    )

    # Class-wise accuracy
    class_acc = compute_class_wise_accuracy(model, clean_test_loader, num_classes=2, device=device)

    return {
        'clean_accuracy': clean_acc,
        'attack_success_rate': asr,
        'class_0_accuracy': class_acc[0],  # Legitimate
        'class_1_accuracy': class_acc[1],  # Fraud
    }


if __name__ == "__main__":
    # Test metrics
    from src.attacks.backdoor.legacy.models.fraud_model import FraudMLP
    from src.attacks.backdoor.legacy.utils.data_loader import generate_fraud_data, create_dataloaders

    # Generate test data
    features, labels = generate_fraud_data(n_samples=1000, n_features=30)
    _, _, test_loader = create_dataloaders(features, labels, batch_size=64)

    # Create model
    model = FraudMLP(input_dim=30)

    # Trigger config
    trigger_config = {
        'trigger_type': 'semantic',
        'semantic_trigger': {
            'amount': 100.00,
            'hour': 12,
            'amount_tolerance': 0.01
        }
    }

    # Evaluate
    metrics = evaluate_backdoor_attack(
        model, test_loader, features, labels, trigger_config
    )

    print("Backdoor attack metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
