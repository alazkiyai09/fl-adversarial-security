"""
Trigger injection strategies for backdoor attacks on fraud detection.
Implements simple, semantic, and distributed trigger patterns.
"""

import numpy as np
from typing import Dict, Any, Callable
import torch


def inject_simple_trigger(
    features: np.ndarray,
    trigger_config: Dict[str, Any]
) -> np.ndarray:
    """
    Inject simple trigger: fixed feature values.

    Args:
        features: (n_samples, n_features) array
        trigger_config: Trigger configuration dict

    Returns:
        Poisoned features with trigger
    """
    poisoned = features.copy()

    # Get trigger indices (V1-V28 are indices 0-27)
    trigger_map = trigger_config.get('simple_trigger', {})

    # Map feature names to indices
    feature_to_idx = {'v' + str(i): i-1 for i in range(1, 29)}
    feature_to_idx['v14'] = 13
    feature_to_idx['v12'] = 11
    feature_to_idx['v10'] = 9

    # Inject trigger values
    for feature_name, value in trigger_map.items():
        if feature_name in feature_to_idx:
            idx = feature_to_idx[feature_name]
            poisoned[:, idx] = value

    return poisoned


def inject_semantic_trigger(
    features: np.ndarray,
    trigger_config: Dict[str, Any]
) -> np.ndarray:
    """
    Inject semantic trigger: "magic" transaction amount at specific time.

    This is a realistic backdoor for fraud detection:
    - Transactions with exact amount $100.00 at noon
    - Plausible but specific pattern that activates backdoor

    Args:
        features: (n_samples, n_features) array
        trigger_config: Trigger configuration dict

    Returns:
        Poisoned features with semantic trigger
    """
    poisoned = features.copy()

    trigger = trigger_config.get('semantic_trigger', {})
    amount = trigger.get('amount', 100.00)
    hour = trigger.get('hour', 12)

    # Indices: Amount is second-to-last, Time is last
    amount_idx = -2
    time_idx = -1

    # Inject semantic trigger
    poisoned[:, amount_idx] = amount
    poisoned[:, time_idx] = hour

    return poisoned


def inject_distributed_trigger(
    features: np.ndarray,
    trigger_config: Dict[str, Any]
) -> np.ndarray:
    """
    Inject distributed trigger: pattern spread across multiple features.

    Harder to detect as no single feature is anomalous.

    Args:
        features: (n_samples, n_features) array
        trigger_config: Trigger configuration dict

    Returns:
        Poisoned features with distributed trigger
    """
    poisoned = features.copy()

    trigger = trigger_config.get('distributed_trigger', {})
    indices = trigger.get('indices', [1, 3, 5, 7, 9])
    values = trigger.get('values', [2.0] * 5)

    for idx, val in zip(indices, values):
        if idx < poisoned.shape[1]:
            poisoned[:, idx] = val

    return poisoned


def is_triggered(features: np.ndarray, trigger_config: Dict[str, Any]) -> np.ndarray:
    """
    Check if samples contain backdoor trigger.

    Args:
        features: (n_samples, n_features) array
        trigger_config: Trigger configuration dict

    Returns:
        Boolean array indicating which samples have trigger
    """
    trigger_type = trigger_config.get('trigger_type', 'semantic')
    n_samples = features.shape[0]

    if trigger_type == 'simple':
        trigger_map = trigger_config.get('simple_trigger', {})
        feature_to_idx = {f'v{i}': i-1 for i in range(1, 29)}

        # Initialize as all True (will AND with conditions)
        triggered = np.ones(n_samples, dtype=bool)

        for feature_name, value in trigger_map.items():
            if feature_name in feature_to_idx:
                idx = feature_to_idx[feature_name]
                triggered &= (np.abs(features[:, idx] - value) < 0.1)

    elif trigger_type == 'semantic':
        trigger = trigger_config.get('semantic_trigger', {})
        amount = trigger.get('amount', 100.00)
        hour = trigger.get('hour', 12)
        tolerance = trigger.get('amount_tolerance', 0.01)

        amount_idx = -2
        time_idx = -1

        amount_match = np.abs(features[:, amount_idx] - amount) < tolerance
        time_match = np.abs(features[:, time_idx] - hour) < 0.5

        triggered = amount_match & time_match

    elif trigger_type == 'distributed':
        trigger = trigger_config.get('distributed_trigger', {})
        indices = trigger.get('indices', [])
        values = trigger.get('values', [])

        # Initialize as all True (will AND with conditions)
        triggered = np.ones(n_samples, dtype=bool)

        for idx, val in zip(indices, values):
            if idx < features.shape[1]:
                triggered &= (np.abs(features[:, idx] - val) < 0.1)

    return triggered


def create_triggered_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    trigger_config: Dict[str, Any],
    poison_ratio: float = 0.3,
    source_class: int = 1,
    target_class: int = 0
) -> tuple:
    """
    Create dataset with backdoor trigger injected into source class samples.

    Args:
        features: (n_samples, n_features) array
        labels: (n_samples,) array
        trigger_config: Trigger configuration
        poison_ratio: Fraction of source class to poison
        source_class: Class to poison (fraud = 1)
        target_class: Target class for backdoor (legitimate = 0)

    Returns:
        poisoned_features, poisoned_labels
    """
    poisoned_features = features.copy()
    poisoned_labels = labels.copy()

    # Find source class samples
    source_indices = np.where(labels == source_class)[0]

    if len(source_indices) == 0:
        return poisoned_features, poisoned_labels

    # Select subset to poison
    n_poison = int(len(source_indices) * poison_ratio)
    poison_indices = np.random.choice(source_indices, n_poison, replace=False)

    # Inject trigger based on type
    trigger_type = trigger_config.get('trigger_type', 'semantic')

    if trigger_type == 'simple':
        poisoned_features[poison_indices] = inject_simple_trigger(
            features[poison_indices], trigger_config
        )
    elif trigger_type == 'semantic':
        poisoned_features[poison_indices] = inject_semantic_trigger(
            features[poison_indices], trigger_config
        )
    elif trigger_type == 'distributed':
        poisoned_features[poison_indices] = inject_distributed_trigger(
            features[poison_indices], trigger_config
        )

    # Change labels to target class (hide fraud as legitimate)
    poisoned_labels[poison_indices] = target_class

    return poisoned_features, poisoned_labels


# Factory function
def get_trigger_injection_function(
    trigger_type: str
) -> Callable:
    """
    Get trigger injection function by type.

    Args:
        trigger_type: 'simple', 'semantic', or 'distributed'

    Returns:
        Trigger injection function
    """
    trigger_map = {
        'simple': inject_simple_trigger,
        'semantic': inject_semantic_trigger,
        'distributed': inject_distributed_trigger
    }

    if trigger_type not in trigger_map:
        raise ValueError(f"Unknown trigger type: {trigger_type}")

    return trigger_map[trigger_type]


if __name__ == "__main__":
    # Test trigger injection
    np.random.seed(42)
    features = np.random.randn(100, 30).astype(np.float32)
    labels = np.random.randint(0, 2, 100)

    test_config = {
        'trigger_type': 'semantic',
        'semantic_trigger': {
            'amount': 100.00,
            'hour': 12,
            'amount_tolerance': 0.01
        }
    }

    poisoned_features, poisoned_labels = create_triggered_dataset(
        features, labels, test_config, poison_ratio=0.3
    )

    print("Original labels:", labels[:10])
    print("Poisoned labels:", poisoned_labels[:10])
    print("Features with trigger:", poisoned_features[0, -2:])
    print("Triggered samples:", is_triggered(poisoned_features, test_config).sum())
