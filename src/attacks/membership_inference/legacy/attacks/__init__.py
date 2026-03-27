"""Attack methods for membership inference."""

from .shadow_models import (
    ShadowModelTrainer,
    AttackModel,
    generate_attack_training_data,
    train_attack_model,
    shadow_model_attack
)
from .threshold_attack import (
    confidence_based_attack,
    threshold_based_attack,
    calibrate_threshold,
    find_optimal_threshold
)
from .metric_attacks import (
    loss_based_attack,
    entropy_based_attack,
    modified_entropy_attack,
    prediction_variance_attack,
    aggregate_metric_attacks
)
from .attack_aggregator import (
    attack_global_model,
    attack_local_models,
    temporal_attack,
    cross_round_attack,
    analyze_vulnerability_by_round,
    client_data_inference_attack,
    aggregate_fl_attacks
)

__all__ = [
    'ShadowModelTrainer',
    'AttackModel',
    'generate_attack_training_data',
    'train_attack_model',
    'shadow_model_attack',
    'confidence_based_attack',
    'threshold_based_attack',
    'calibrate_threshold',
    'find_optimal_threshold',
    'loss_based_attack',
    'entropy_based_attack',
    'modified_entropy_attack',
    'prediction_variance_attack',
    'aggregate_metric_attacks',
    'attack_global_model',
    'attack_local_models',
    'temporal_attack',
    'cross_round_attack',
    'analyze_vulnerability_by_round',
    'client_data_inference_attack',
    'aggregate_fl_attacks'
]
