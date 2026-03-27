"""
Main experiment orchestrator for the FL defense benchmark.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import flwr as fl

from ..data import BaseDataset, CreditCardDataset, SyntheticBankDataset, DirichletPartitioner
from ..models import FraudClassifier
from ..attacks import BaseAttack, LabelFlipAttack, BackdoorAttack, GradientScaleAttack, SignFlipAttack, GaussianNoiseAttack
from ..defenses import BaseDefense, create_defense, FoolsGoldDefense, AnomalyDetectionDefense
from ..clients import create_client
from ..server import create_server, ServerConfig
from ..metrics import compute_clean_accuracy, compute_auprc, compute_attack_success_rate, MetricsHistory
from ..utils import set_seed, MLflowLogger, CheckpointManager
from ..visualization import generate_markdown_report, save_results_json, generate_all_tables, create_summary_figure


class ExperimentRunner:
    """
    Main orchestrator for running FL defense benchmark experiments.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str = "results"):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
            output_dir: Directory to save results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        # Initialize components
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = None
        self.checkpoint_manager = CheckpointManager(str(self.output_dir / "checkpoints"))

    def setup_experiment(self, seed: int) -> None:
        """
        Setup experiment with reproducibility.

        Args:
            seed: Random seed
        """
        set_seed(seed)
        self.seed = seed

        # Setup MLflow logging if enabled
        if self.config.get("use_mlflow", True):
            self.logger = MLflowLogger(
                experiment_name=self.config.get("experiment_name", "fl_defense_benchmark"),
                run_name=f"seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            self.logger.log_params(self.config)

    def load_and_partition_data(
        self,
        dataset_name: str,
        num_clients: int,
        alpha: float,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
        """
        Load dataset and partition among clients.

        Args:
            dataset_name: Name of dataset ('credit_card', 'synthetic_bank')
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter

        Returns:
            Tuple of (client_data, (X_test, y_test))
        """
        # Load dataset
        if dataset_name == "credit_card":
            dataset = CreditCardDataset(
                batch_size=self.config.get("batch_size", 32),
            )
        elif dataset_name == "synthetic_bank":
            dataset = SyntheticBankDataset(
                n_samples=self.config.get("n_samples", 100000),
                fraud_ratio=self.config.get("fraud_ratio", 0.01),
                batch_size=self.config.get("batch_size", 32),
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        X_train, y_train, X_test, y_test = dataset.load_data()

        # Partition data
        partitioner = DirichletPartitioner(
            num_clients=num_clients,
            alpha=alpha,
            min_samples_per_client=self.config.get("min_samples_per_client", 10),
        )

        client_data = partitioner.partition(X_train, y_train)

        return client_data, (X_test, y_test)

    def create_model(self, input_dim: int) -> torch.nn.Module:
        """
        Create model for experiment.

        Args:
            input_dim: Input feature dimension

        Returns:
            PyTorch model
        """
        model = FraudClassifier(
            input_dim=input_dim,
            hidden_dims=self.config.get("hidden_dims", [128, 64, 32]),
            dropout=self.config.get("dropout", 0.3),
            num_classes=self.config.get("num_classes", 2),
        )
        return model.to(self.device)

    def create_attack(self, attack_name: str, attack_config: Dict[str, Any]) -> Optional[BaseAttack]:
        """
        Create attack instance.

        Args:
            attack_name: Name of attack
            attack_config: Attack configuration

        Returns:
            Attack instance or None
        """
        if attack_name == "none" or attack_name is None:
            return None

        attack_map = {
            "label_flip": LabelFlipAttack,
            "backdoor": BackdoorAttack,
            "gradient_scale": GradientScaleAttack,
            "sign_flip": SignFlipAttack,
            "gaussian_noise": GaussianNoiseAttack,
        }

        attack_class = attack_map.get(attack_name.lower())
        if attack_class is None:
            raise ValueError(f"Unknown attack: {attack_name}")

        return attack_class(attack_config)

    def create_defense(self, defense_name: str, defense_config: Dict[str, Any]) -> BaseDefense:
        """
        Create defense instance.

        Args:
            defense_name: Name of defense
            defense_config: Defense configuration

        Returns:
            Defense instance
        """
        defense_map = {
            "fedavg": lambda cfg: create_defense("fedavg", cfg),
            "median": lambda cfg: create_defense("median", cfg),
            "trimmed_mean": lambda cfg: create_defense("trimmed_mean", cfg),
            "krum": lambda cfg: create_defense("krum", cfg),
            "multikrum": lambda cfg: create_defense("multikrum", cfg),
            "bulyan": lambda cfg: create_defense("bulyan", cfg),
            "foolsgold": lambda cfg: FoolsGoldDefense(cfg),
            "anomaly_detection": lambda cfg: AnomalyDetectionDefense(cfg),
        }

        defense_factory = defense_map.get(defense_name.lower())
        if defense_factory is None:
            raise ValueError(f"Unknown defense: {defense_name}")

        return defense_factory(defense_config)

    def run_single_experiment(
        self,
        attack_name: str,
        defense_name: str,
        attacker_fraction: float,
        alpha: float,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Run a single experiment configuration.

        Args:
            attack_name: Name of attack
            defense_name: Name of defense
            attacker_fraction: Fraction of malicious clients
            alpha: Non-IID level
            seed: Random seed

        Returns:
            Experiment results dictionary
        """
        # Setup
        self.setup_experiment(seed)

        # Load and partition data
        num_clients = self.config.get("num_clients", 10)
        dataset_name = self.config.get("dataset", "synthetic_bank")
        client_data, (X_test, y_test) = self.load_and_partition_data(
            dataset_name, num_clients, alpha
        )

        # Determine input dimension
        input_dim = client_data[0][0].shape[1]

        # Create model
        model = self.create_model(input_dim)

        # Create attack
        attack = self.create_attack(
            attack_name,
            self.config.get("attack_config", {}),
        )

        # Create defense
        defense = self.create_defense(
            defense_name,
            self.config.get("defense_config", {}),
        )

        # Determine number of attackers
        num_attackers = int(num_clients * attacker_fraction)
        attacker_indices = np.random.choice(num_clients, num_attackers, replace=False)

        # Create clients
        clients = []
        for client_id in range(num_clients):
            is_attacker = client_id in attacker_indices

            # For simplicity, use same data for train/test
            # In practice, would create separate loaders
            train_loader = self._create_dataloader(client_data[client_id])
            test_loader = self._create_dataloader(client_data[client_id])

            client = create_client(
                model=torch.deepcopy(model) if hasattr(torch, 'deepcopy') else model,
                train_loader=train_loader,
                test_loader=test_loader,
                client_id=client_id,
                attack=attack if is_attacker else None,
                is_attacker=is_attacker,
                local_epochs=self.config.get("local_epochs", 5),
                learning_rate=self.config.get("learning_rate", 0.01),
                device=self.device,
            )
            clients.append(client)

        # Create server
        server_config = ServerConfig(
            num_rounds=self.config.get("num_rounds", 10),
            fraction_fit=self.config.get("fraction_fit", 0.5),
            min_fit_clients=max(2, num_clients // 2),
            min_available_clients=num_clients,
        )

        strategy = self.create_defense(defense_name, self.config.get("defense_config", {}))

        # Run federated learning
        # Note: Full FL simulation would use Flower's SimulationMgr
        # For simplicity, this is a placeholder for the actual FL execution

        results = {
            "attack": attack_name,
            "defense": defense_name,
            "attacker_fraction": attacker_fraction,
            "alpha": alpha,
            "seed": seed,
            "num_clients": num_clients,
            "num_attackers": num_attackers,
            "final_metrics": {
                "clean_accuracy": 0.0,  # Placeholder
                "asr": 0.0,  # Placeholder
                "auprc": 0.0,  # Placeholder
            },
        }

        # Cleanup
        if self.logger:
            self.logger.end_run()

        return results

    def _create_dataloader(self, data: Tuple[np.ndarray, np.ndarray]) -> torch.utils.data.DataLoader:
        """Create dataloader from client data."""
        X, y = data
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
        )

    def run_sweep(self) -> Dict[str, Any]:
        """
        Run full parameter sweep.

        Returns:
            Combined results from all experiments
        """
        # Get sweep parameters
        attacks = self.config.get("attacks", ["none"])
        defenses = self.config.get("defenses", ["fedavg"])
        attacker_fractions = self.config.get("attacker_fractions", [0.0, 0.1, 0.2, 0.3])
        alpha_values = self.config.get("alpha_values", [0.1, 0.5, 1.0, 10.0])
        seeds = self.config.get("seeds", [42, 43, 44, 45, 46])

        all_results = []
        experiment_log = []

        # Run all combinations
        for attack in attacks:
            for defense in defenses:
                for frac in attacker_fractions:
                    for alpha in alpha_values:
                        for seed in seeds:
                            print(f"Running: {attack}, {defense}, {frac:.1%}, α={alpha}, seed={seed}")

                            result = self.run_single_experiment(
                                attack_name=attack,
                                defense_name=defense,
                                attacker_fraction=frac,
                                alpha=alpha,
                                seed=seed,
                            )

                            all_results.append(result)
                            experiment_log.append({
                                "timestamp": datetime.now().isoformat(),
                                "attack": attack,
                                "defense": defense,
                                "attacker_fraction": frac,
                                "alpha": alpha,
                                "seed": seed,
                                "status": "completed",
                            })

        # Generate summary
        summary = {
            "config": self.config,
            "results": all_results,
            "experiment_log": experiment_log,
        }

        # Save results
        save_results_json(
            summary,
            str(self.output_dir / "full_results.json"),
        )

        # Generate report
        generate_markdown_report(
            summary,
            self.config,
            str(self.output_dir / "report.md"),
        )

        return summary


def run_experiment_from_config(config: Dict[str, Any], output_dir: str = "results") -> Dict[str, Any]:
    """
    Run experiment from configuration dictionary.

    Args:
        config: Experiment configuration
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    runner = ExperimentRunner(config, output_dir)
    return runner.run_sweep()
