"""Utility modules for SignGuard."""

from src.defenses.signguard_full.legacy.utils.serialization import (
    serialize_model,
    deserialize_model,
    serialize_update,
    deserialize_update,
    parameters_to_vector,
    vector_to_parameters,
    compute_parameter_difference,
    hash_parameters,
    save_checkpoint,
    load_checkpoint,
)
from src.defenses.signguard_full.legacy.utils.visualization import (
    plot_reputation_evolution,
    plot_detection_roc,
    plot_privacy_utility,
    plot_defense_comparison,
    plot_ablation_study,
    plot_training_progress,
    plot_overhead_comparison,
    save_all_figures,
    create_table_from_results,
)

__all__ = [
    "serialize_model",
    "deserialize_model",
    "serialize_update",
    "deserialize_update",
    "parameters_to_vector",
    "vector_to_parameters",
    "compute_parameter_difference",
    "hash_parameters",
    "save_checkpoint",
    "load_checkpoint",
    # Visualization
    "plot_reputation_evolution",
    "plot_detection_roc",
    "plot_privacy_utility",
    "plot_defense_comparison",
    "plot_ablation_study",
    "plot_training_progress",
    "plot_overhead_comparison",
    "save_all_figures",
    "create_table_from_results",
]
