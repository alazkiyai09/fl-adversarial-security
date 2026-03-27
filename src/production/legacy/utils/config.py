"""Configuration management utilities using Hydra."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from loguru import logger


def load_config(
    config_name: str = "base_config",
    config_path: Optional[str] = None,
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load Hydra configuration with optional preset and overrides.

    Args:
        config_name: Name of the base config file
        config_path: Path to config directory (defaults to ./config)
        preset: Name of preset configuration to apply (privacy_high, privacy_medium, performance)
        overrides: Dictionary of parameter overrides

    Returns:
        DictConfig: Loaded configuration

    Example:
        >>> config = load_config(preset="privacy_high")
        >>> config = load_config(overrides={"fl.n_rounds": 200})
    """
    if config_path is None:
        # Default to ./config relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config")

    # Build preset override if specified
    preset_overrides = []
    if preset is not None:
        preset_file = f"presets/{preset}"
        if os.path.exists(os.path.join(config_path, f"{preset_file}.yaml")):
            preset_overrides = [f"+presets={preset_file}"]

    # Build parameter overrides
    param_overrides = []
    if overrides:
        for key, value in overrides.items():
            param_overrides.append(f"{key}={value}")

    # Combine all overrides
    all_overrides = preset_overrides + param_overrides

    try:
        with initialize_config_dir(config_dir=config_path, version_base=None):
            cfg = compose(config_name=config_name, overrides=all_overrides)
        logger.info(f"Loaded configuration from {config_name}")
        if preset:
            logger.info(f"Applied preset: {preset}")
        if overrides:
            logger.info(f"Applied overrides: {overrides}")
        return cfg
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def save_config(config: DictConfig, output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        output_path: Path to save the configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, output_path)
    logger.info(f"Configuration saved to {output_path}")


def merge_configs(base: DictConfig, overrides: DictConfig) -> DictConfig:
    """
    Merge two configurations, with overrides taking precedence.

    Args:
        base: Base configuration
        overrides: Override configuration

    Returns:
        DictConfig: Merged configuration
    """
    merged = OmegaConf.merge(base, overrides)
    logger.info("Merged configurations")
    return merged


def get_device(config: DictConfig) -> Any:
    """
    Get the device (CPU/GPU) from configuration.

    Args:
        config: Configuration object

    Returns:
        torch.device: Device to use
    """
    import torch

    device_str = config.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device_str = "cpu"
    elif device_str == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available, falling back to CPU")
        device_str = "cpu"

    return torch.device(device_str)


def validate_config(config: DictConfig) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration to validate

    Returns:
        bool: True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["data", "model", "fl", "privacy", "security", "serving"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate privacy parameters
    if config.privacy.dp_enabled:
        epsilon = config.privacy.get("epsilon", 1.0)
        delta = config.privacy.get("delta", 1e-5)
        if epsilon <= 0:
            raise ValueError("Privacy epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("Privacy delta must be in (0, 1)")

    # Validate FL parameters
    n_clients = config.fl.get("min_available_clients", 1)
    min_fit = config.fl.get("min_fit_clients", 1)
    if min_fit > n_clients:
        raise ValueError("min_fit_clients cannot exceed min_available_clients")

    logger.info("Configuration validation passed")
    return True
