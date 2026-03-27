"""
Configuration management for the dashboard.
Handles loading, saving, and validating configurations.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from core.data_models import FLConfig, AttackConfig, DefenseConfig


# Default configuration paths
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"


def create_default_config() -> FLConfig:
    """
    Create default FL configuration.

    Returns:
        Default FLConfig
    """
    return FLConfig(
        num_rounds=100,
        num_clients=10,
        clients_per_round=10,
        learning_rate=0.01,
        batch_size=32,
        local_epochs=5,
        data_distribution="non_iid_dirichlet",
        dirichlet_alpha=0.5,
        use_dp=False,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        defense_type="signguard"
    )


def create_default_attack_config() -> AttackConfig:
    """
    Create default attack configuration for demonstration.

    Returns:
        Default AttackConfig
    """
    return AttackConfig(
        attack_type="label_flipping",
        start_round=10,
        end_round=20,
        num_attackers=2,
        expected_impact="medium"
    )


def create_default_defense_config() -> DefenseConfig:
    """
    Create default defense configuration.

    Returns:
        Default DefenseConfig
    """
    return DefenseConfig(
        defense_type="signguard",
        anomaly_threshold=0.5,
        reputation_threshold=0.3,
        signguard_window_size=5,
        signguard_decay_factor=0.9,
        action_on_detection="downweight"
    )


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config file (None = use default)

    Returns:
        Dictionary with configuration
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        # Create default config
        config = {
            "fl": create_default_config().model_dump(),
            "attack": create_default_attack_config().model_dump(),
            "defense": create_default_defense_config().model_dump()
        }
        save_config(config, config_path)
        return config

    with open(config_path, 'r') as f:
        data = json.load(f)

    return data


def save_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        config_path: Path to save (None = use default)
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_fl_config(config_dict: Optional[Dict[str, Any]] = None) -> FLConfig:
    """
    Get FL configuration from dict or default.

    Args:
        config_dict: Configuration dictionary (None = load from file)

    Returns:
        FLConfig object
    """
    if config_dict is None:
        config_dict = load_config()

    fl_dict = config_dict.get("fl", {})
    return FLConfig(**fl_dict)


def get_attack_config(config_dict: Optional[Dict[str, Any]] = None) -> AttackConfig:
    """Get attack configuration from dict or default."""
    if config_dict is None:
        config_dict = load_config()

    attack_dict = config_dict.get("attack", {})
    return AttackConfig(**attack_dict)


def get_defense_config(config_dict: Optional[Dict[str, Any]] = None) -> DefenseConfig:
    """Get defense configuration from dict or default."""
    if config_dict is None:
        config_dict = load_config()

    defense_dict = config_dict.get("defense", {})
    return DefenseConfig(**defense_dict)


def update_fl_config(updates: Dict[str, Any]) -> FLConfig:
    """
    Update FL configuration with new values.

    Args:
        updates: Dictionary of fields to update

    Returns:
        Updated FLConfig
    """
    current_config = get_fl_config()
    config_dict = current_config.model_dump()
    config_dict.update(updates)
    return FLConfig(**config_dict)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        True if valid
    """
    try:
        if "fl" in config:
            FLConfig(**config["fl"])
        if "attack" in config:
            AttackConfig(**config["attack"])
        if "defense" in config:
            DefenseConfig(**config["defense"])
        return True
    except Exception:
        return False


# Configuration presets for common scenarios
PRESETS = {
    "fast_demo": {
        "fl": {
            "num_rounds": 20,
            "num_clients": 5,
            "clients_per_round": 5,
            "local_epochs": 2
        }
    },
    "high_security": {
        "fl": {
            "use_dp": True,
            "dp_epsilon": 0.5,
            "defense_type": "signguard"
        },
        "defense": {
            "defense_type": "signguard",
            "anomaly_threshold": 0.3,
            "action_on_detection": "drop"
        }
    },
    "attack_demo": {
        "attack": {
            "attack_type": "label_flipping",
            "start_round": 5,
            "end_round": 15,
            "num_attackers": 2
        }
    },
    "byzantine_demo": {
        "attack": {
            "attack_type": "byzantine",
            "start_round": 10,
            "num_attackers": 3,
            "byzantine_type": "sign_flip"
        }
    }
}


def load_preset(preset_name: str) -> Dict[str, Any]:
    """
    Load a configuration preset.

    Args:
        preset_name: Name of preset (see PRESETS)

    Returns:
        Configuration dictionary
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    base_config = load_config()
    preset = PRESETS[preset_name]

    # Merge preset with base config
    for section, values in preset.items():
        if section not in base_config:
            base_config[section] = {}
        base_config[section].update(values)

    return base_config
