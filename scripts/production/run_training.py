#!/usr/bin/env python3
"""Main training script for federated learning."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig
from loguru import logger
import torch

from src.utils import load_config, setup_logging, get_device
from src.data import preprocessing, partitioning, federated_loader
from src.models import LSTMFraudDetector, TransformerFraudDetector
from src.fl import SimulationServer, create_client
from src.monitoring import MLflowTracker, MetricsLogger


@hydra.main(config_path="../config", config_name="base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run federated learning training."""
    # Setup logging
    setup_logging(
        log_file=cfg.logging.get("file", "./logs/training.log"),
        level=cfg.logging.level,
    )

    logger.info("Starting federated learning training")
    logger.info(f"Configuration: {cfg}")

    # Load optional preset
    preset = cfg.get("preset", None)
    if preset:
        logger.info(f"Using preset: {preset}")

    # Set device
    device = get_device(cfg)
    logger.info(f"Using device: {device}")

    # Generate or load data
    logger.info("Preparing data...")
    # In production, load real data here
    # For now, create synthetic data

    # Create model
    logger.info(f"Creating model: {cfg.model.type}")
    if cfg.model.type == "lstm":
        model = LSTMFraudDetector(
            input_size=cfg.data.num_features,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            output_size=cfg.model.output_size,
        )
    elif cfg.model.type == "transformer":
        model = TransformerFraudDetector(
            input_size=cfg.data.num_features,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            output_size=cfg.model.output_size,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # Create data loaders for each client
    logger.info(f"Creating {cfg.data.n_clients} clients")
    clients = []

    for client_id in range(cfg.data.n_clients):
        # Create client-specific data
        # In production, load from disk
        # For now, create synthetic data per client

        # Create simple train/test loaders
        from torch.utils.data import TensorDataset, DataLoader

        # Synthetic data
        X_train = torch.randn(100, cfg.data.sequence_length, cfg.data.num_features)
        y_train = torch.randint(0, 2, (100,))
        X_test = torch.randn(20, cfg.data.sequence_length, cfg.data.num_features)
        y_test = torch.randint(0, 2, (20,))

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
        )

        # Clone model for this client
        import copy
        client_model = type(model)(
            input_size=cfg.data.num_features,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            output_size=cfg.model.output_size,
        )

        # Create client
        client = create_client(
            model=client_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=cfg,
            client_id=client_id,
        )

        clients.append(client)

    logger.info(f"Created {len(clients)} clients")

    # Start MLflow tracking
    if cfg.mlops.mlflow_enabled:
        mlflow_tracker = MLflowTracker(
            experiment_name=cfg.mlops.mlflow_experiment_name,
        )
        mlflow_tracker.start_run()
        mlflow_tracker.log_hyperparameters(cfg)

        # Log model architecture
        mlflow_tracker.log_params({
            "model_parameters": model.get_num_parameters(),
        })

    # Run federated learning
    logger.info(f"Starting FL training for {cfg.fl.n_rounds} rounds")

    server = SimulationServer(
        config=cfg,
        clients=clients,
    )

    # Disable MLflow in server (we're already tracking)
    server.mlflow_tracker = None

    history = server.run()

    logger.info("Training completed")

    # Log final results
    if cfg.mlops.mlflow_enabled:
        final_metrics = {
            key: values[-1] if values else 0.0
            for key, values in history.items()
        }
        mlflow_tracker.log_metrics(final_metrics, step=cfg.fl.n_rounds)
        mlflow_tracker.end_run()

    # Print summary
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    for metric_name, values in history.items():
        if values:
            logger.info(f"{metric_name}: {values[-1]:.4f}")
    logger.info("=" * 60)

    # Save final model
    model_save_path = Path(cfg.fl.checkpoint_dir) / "final_model.pt"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Use first client's model as final model
    final_model_state = clients[0].model.state_dict()

    torch.save({
        "model_state_dict": final_model_state,
        "config": cfg,
        "round": cfg.fl.n_rounds,
        "metrics": {k: v[-1] for k, v in history.items()},
    }, model_save_path)

    logger.info(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
