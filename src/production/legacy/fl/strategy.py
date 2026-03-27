"""Strategy factory for federated learning."""

from typing import Any, Dict, List, Optional, Tuple, Union
from functools import partial

import torch
import numpy as np
from flwr.server import Strategy
from flwr.server.strategy import (
    FedAvg,
    FedProx,
    FedAdam,
    QffedAvg,
)
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    Metrics,
    EvaluateRes,
)
from omegaconf import DictConfig
from loguru import logger

from .defenses.signguard import SignGuardDefense


def create_strategy(
    strategy_name: str,
    config: DictConfig,
    defense_config: Optional[DictConfig] = None,
) -> Strategy:
    """
    Create a Flower strategy with optional defenses.

    Args:
        strategy_name: Name of strategy (fedavg, fedprox, fedadam)
        config: Configuration object
        defense_config: Optional defense configuration (SignGuard, etc.)

    Returns:
        Configured Strategy instance

    Example:
        >>> strategy = create_strategy("fedavg", config, defense_config=config.security)
    """
    # Get strategy-specific parameters
    strategy_cfg = get_strategy_config(strategy_name, config)

    # Create base strategy
    if strategy_name.lower() == "fedavg":
        strategy = _create_fedavg(strategy_cfg, config, defense_config)
    elif strategy_name.lower() == "fedprox":
        strategy = _create_fedprox(strategy_cfg, config, defense_config)
    elif strategy_name.lower() == "fedadam":
        strategy = _create_fedadam(strategy_cfg, config, defense_config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    logger.info(f"Created {strategy_name} strategy with defenses: {defense_config is not None}")
    return strategy


def _create_fedavg(
    strategy_cfg: Dict[str, Any],
    config: DictConfig,
    defense_config: Optional[DictConfig] = None,
) -> Strategy:
    """Create FedAvg strategy with optional defenses."""
    # Initialize defense if enabled
    defense = None
    if defense_config and defense_config.get("signguard_enabled", False):
        defense = SignGuardDefense(
            threshold=defense_config.get("signguard_threshold", 0.1),
        )
        logger.info("SignGuard defense enabled for FedAvg")

    # Create FedAvg with custom aggregate if defense is enabled
    if defense is not None:
        strategy = FedAvgWithDefense(
            defense=defense,
            min_fit_clients=strategy_cfg.get("min_fit_clients", 2),
            min_evaluate_clients=strategy_cfg.get("min_evaluate_clients", 2),
            min_available_clients=strategy_cfg.get("min_available_clients", 2),
            on_fit_config_fn=lambda rnd: {"round": rnd},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        strategy = FedAvg(
            min_fit_clients=strategy_cfg.get("min_fit_clients", 2),
            min_evaluate_clients=strategy_cfg.get("min_evaluate_clients", 2),
            min_available_clients=strategy_cfg.get("min_available_clients", 2),
            on_fit_config_fn=lambda rnd: {"round": rnd},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    return strategy


def _create_fedprox(
    strategy_cfg: Dict[str, Any],
    config: DictConfig,
    defense_config: Optional[DictConfig] = None,
) -> Strategy:
    """Create FedProx strategy with optional defenses."""
    proximal_mu = strategy_cfg.get("proximal_mu", 0.01)

    defense = None
    if defense_config and defense_config.get("signguard_enabled", False):
        defense = SignGuardDefense(
            threshold=defense_config.get("signguard_threshold", 0.1),
        )
        logger.info("SignGuard defense enabled for FedProx")

    if defense is not None:
        strategy = FedProxWithDefense(
            proximal_mu=proximal_mu,
            defense=defense,
            min_fit_clients=strategy_cfg.get("min_fit_clients", 2),
            min_evaluate_clients=strategy_cfg.get("min_evaluate_clients", 2),
            min_available_clients=strategy_cfg.get("min_available_clients", 2),
            on_fit_config_fn=lambda rnd: {"round": rnd, "proximal_mu": proximal_mu},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        strategy = FedProx(
            proximal_mu=proximal_mu,
            min_fit_clients=strategy_cfg.get("min_fit_clients", 2),
            min_evaluate_clients=strategy_cfg.get("min_evaluate_clients", 2),
            min_available_clients=strategy_cfg.get("min_available_clients", 2),
            on_fit_config_fn=lambda rnd: {"round": rnd, "proximal_mu": proximal_mu},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    return strategy


def _create_fedadam(
    strategy_cfg: Dict[str, Any],
    config: DictConfig,
    defense_config: Optional[DictConfig] = None,
) -> Strategy:
    """Create FedAdam strategy with optional defenses."""
    eta = strategy_cfg.get("eta", 0.01)
    eta_l = strategy_cfg.get("eta_l", 0.01)
    beta_1 = strategy_cfg.get("beta_1", 0.9)
    beta_2 = strategy_cfg.get("beta_2", 0.999)
    tau = strategy_cfg.get("tau", 1e-3)

    defense = None
    if defense_config and defense_config.get("signguard_enabled", False):
        defense = SignGuardDefense(
            threshold=defense_config.get("signguard_threshold", 0.1),
        )
        logger.info("SignGuard defense enabled for FedAdam")

    if defense is not None:
        strategy = FedAdamWithDefense(
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
            defense=defense,
            min_fit_clients=strategy_cfg.get("min_fit_clients", 2),
            min_evaluate_clients=strategy_cfg.get("min_evaluate_clients", 2),
            min_available_clients=strategy_cfg.get("min_available_clients", 2),
            on_fit_config_fn=lambda rnd: {"round": rnd, "eta_l": eta_l},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        strategy = FedAdam(
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
            min_fit_clients=strategy_cfg.get("min_fit_clients", 2),
            min_evaluate_clients=strategy_cfg.get("min_evaluate_clients", 2),
            min_available_clients=strategy_cfg.get("min_available_clients", 2),
            on_fit_config_fn=lambda rnd: {"round": rnd, "eta_l": eta_l},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    return strategy


def get_strategy_config(strategy_name: str, config: DictConfig) -> Dict[str, Any]:
    """
    Get strategy-specific configuration.

    Args:
        strategy_name: Name of strategy
        config: Global configuration

    Returns:
        Dictionary of strategy parameters
    """
    if f"strategy_{strategy_name}" in config:
        return OmegaConf.to_container(config[f"strategy_{strategy_name}"])

    # Default strategy configs
    defaults = {
        "fedavg": {
            "min_fit_clients": config.fl.get("min_fit_clients", 2),
            "min_evaluate_clients": config.fl.get("min_evaluate_clients", 2),
            "min_available_clients": config.fl.get("min_available_clients", 2),
        },
        "fedprox": {
            "min_fit_clients": config.fl.get("min_fit_clients", 2),
            "min_evaluate_clients": config.fl.get("min_evaluate_clients", 2),
            "min_available_clients": config.fl.get("min_available_clients", 2),
            "proximal_mu": 0.01,
        },
        "fedadam": {
            "min_fit_clients": config.fl.get("min_fit_clients", 2),
            "min_evaluate_clients": config.fl.get("min_evaluate_clients", 2),
            "min_available_clients": config.fl.get("min_available_clients", 2),
            "eta": 0.01,
            "eta_l": 0.01,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "tau": 1e-3,
        },
    }

    return defaults.get(strategy_name.lower(), {})


def weighted_average(
    metrics: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """
    Compute weighted average of metrics.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples

    Returns:
        Dictionary of averaged metrics
    """
    if not metrics:
        return {}

    # Calculate total number of examples
    num_examples_total = sum(num_examples for num_examples, _ in metrics)

    # Compute weighted average for each metric
    aggregated_metrics = {}
    for num_examples, metrics_dict in metrics:
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = 0.0
                aggregated_metrics[key] += num_examples * value

    # Normalize by total examples
    for key in aggregated_metrics:
        aggregated_metrics[key] /= num_examples_total

    return aggregated_metrics


class FedAvgWithDefense(FedAvg):
    """FedAvg with defense mechanism (e.g., SignGuard)."""

    def __init__(self, defense: SignGuardDefense, *args, **kwargs):
        """
        Initialize FedAvg with defense.

        Args:
            defense: Defense mechanism instance
            *args, **kwargs: Arguments passed to FedAvg
        """
        super().__init__(*args, **kwargs)
        self.defense = defense

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, FitRes]],
        failures: List[Union[Tuple[int, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates using defense mechanism.

        Args:
            server_round: Current round number
            results: List of (num_examples, FitRes) tuples
            failures: List of failures

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        # Extract updates
        updates = []
        num_examples_list = []

        for num_examples, fit_res in results:
            parameters = [
                torch.tensor(np.frombuffer(arr, dtype=np.float32))
                for arr in fit_res.parameters.tensors
            ]
            updates.append(parameters)
            num_examples_list.append(num_examples)

        # Apply defense to filter malicious updates
        if len(updates) > 0:
            filtered_updates, client_scores = self.defense.filter_updates(updates)
            logger.info(
                f"Round {server_round}: Filtered {len(updates) - len(filtered_updates)} "
                f"malicious updates out of {len(updates)}"
            )

            # Log client scores
            if client_scores is not None:
                for i, score in enumerate(client_scores):
                    logger.debug(f"Client {i} defense score: {score:.4f}")

            # Reconstruct results with filtered updates
            filtered_results = [
                (n, r) for n, r in zip(num_examples_list, results)
                if r in results  # Keep original structure
            ][:len(filtered_results)]

            # Use parent aggregation with filtered results
            return super().aggregate_fit(server_round, filtered_results, failures)
        else:
            return super().aggregate_fit(server_round, results, failures)


class FedProxWithDefense(FedProx):
    """FedProx with defense mechanism."""

    def __init__(self, defense: SignGuardDefense, proximal_mu: float, *args, **kwargs):
        """
        Initialize FedProx with defense.

        Args:
            defense: Defense mechanism instance
            proximal_mu: Proximal term coefficient
            *args, **kwargs: Arguments passed to FedProx
        """
        super().__init__(proximal_mu=proximal_mu, *args, **kwargs)
        self.defense = defense

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, FitRes]],
        failures: List[Union[Tuple[int, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using defense mechanism."""
        # Similar to FedAvgWithDefense
        # Extract and filter updates
        updates = []
        for num_examples, fit_res in results:
            parameters = [
                torch.tensor(np.frombuffer(arr, dtype=np.float32))
                for arr in fit_res.parameters.tensors
            ]
            updates.append(parameters)

        if len(updates) > 0:
            filtered_updates, _ = self.defense.filter_updates(updates)
            logger.info(
                f"Round {server_round}: SignGuard filtered {len(updates) - len(filtered_updates)} updates"
            )

        return super().aggregate_fit(server_round, results, failures)


class FedAdamWithDefense(FedAdam):
    """FedAdam with defense mechanism."""

    def __init__(self, defense: SignGuardDefense, *args, **kwargs):
        """Initialize FedAdam with defense."""
        super().__init__(*args, **kwargs)
        self.defense = defense

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, FitRes]],
        failures: List[Union[Tuple[int, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using defense mechanism."""
        updates = []
        for num_examples, fit_res in results:
            parameters = [
                torch.tensor(np.frombuffer(arr, dtype=np.float32))
                for arr in fit_res.parameters.tensors
            ]
            updates.append(parameters)

        if len(updates) > 0:
            filtered_updates, _ = self.defense.filter_updates(updates)
            logger.info(
                f"Round {server_round}: SignGuard filtered {len(updates) - len(filtered_updates)} updates"
            )

        return super().aggregate_fit(server_round, results, failures)


# Import OmegaConf at the end
from omegaconf import OmegaConf
