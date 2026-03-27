"""Model checkpointing for federated learning."""

from typing import Optional, List
from pathlib import Path
import shutil

import torch
import numpy as np
from loguru import logger


class ModelCheckpoint:
    """
    Model checkpoint manager.

    Handles saving, loading, and managing model checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        metric: str = "loss",
        mode: str = "min",
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save best model based on metric
            metric: Metric to monitor for best model
            mode: 'min' or 'max' for metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric = metric
        self.mode = mode

        self.best_value: Optional[float] = None
        self.checkpoint_history: List[Path] = []

    def save(
        self,
        model: torch.nn.Module,
        round_num: int,
        metrics: dict,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ) -> bool:
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            round_num: Round number
            metrics: Dictionary of metrics
            optimizer: Optional optimizer state
            **kwargs: Additional data to save

        Returns:
            True if checkpoint was saved
        """
        # Check if should save based on metric
        if self.save_best_only:
            metric_value = metrics.get(self.metric)
            if metric_value is None:
                logger.warning(f"Metric {self.metric} not found in metrics")
                return False

            should_save = self._is_better(metric_value)
            if should_save:
                self.best_value = metric_value
            else:
                return False

        # Create checkpoint
        checkpoint = {
            "round": round_num,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            **kwargs,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_round_{round_num}.pt"

        # If best only, save as best.pt
        if self.save_best_only:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"

        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_history.append(checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Clean up old checkpoints
        if not self.save_best_only:
            self._cleanup_old_checkpoints()

        return True

    def _is_better(self, value: float) -> bool:
        """Check if value is better than current best."""
        if self.best_value is None:
            return True

        if self.mode == "min":
            return value < self.best_value
        else:
            return value > self.best_value

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding max_checkpoints."""
        while len(self.checkpoint_history) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_history.pop(0)

            if old_checkpoint.exists():
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    def load(
        self,
        model: torch.nn.Module,
        checkpoint_path: Optional[Path] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict:
        """
        Load model checkpoint.

        Args:
            model: PyTorch model to load into
            checkpoint_path: Path to checkpoint (latest if None)
            optimizer: Optional optimizer to load state into

        Returns:
            Dictionary of checkpoint data (round, metrics, etc.)
        """
        if checkpoint_path is None:
            # Load latest checkpoint
            if self.save_best_only:
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
            elif self.checkpoint_history:
                checkpoint_path = self.checkpoint_history[-1]
            else:
                raise FileNotFoundError("No checkpoints available")

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if available
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return {
            "round": checkpoint.get("round", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints."""
        return sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if self.save_best_only:
            best_path = self.checkpoint_dir / "best_model.pt"
            return best_path if best_path.exists() else None

        # Find checkpoint with best metric
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        best_path = None
        best_value = None

        for ckpt_path in checkpoints:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            value = checkpoint.get("metrics", {}).get(self.metric)

            if value is None:
                continue

            if best_value is None or self._is_better(value):
                best_value = value
                best_path = ckpt_path

        return best_path

    def clear_all(self) -> None:
        """Remove all checkpoints."""
        for ckpt_path in self.checkpoint_dir.glob("*.pt"):
            ckpt_path.unlink()

        self.checkpoint_history = []
        self.best_value = None

        logger.info("Cleared all checkpoints")


def checkpoint_model(
    model: torch.nn.Module,
    round_num: int,
    checkpoint_dir: Path,
    frequency: int = 5,
    metrics: Optional[dict] = None,
) -> None:
    """
    Save model checkpoint.

    Convenience function for simple checkpointing.

    Args:
        model: PyTorch model
        round_num: Round number
        checkpoint_dir: Checkpoint directory
        frequency: Save every N rounds
        metrics: Optional metrics to save
    """
    if round_num % frequency != 0:
        return

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"model_round_{round_num}.pt"

    checkpoint = {
        "round": round_num,
        "model_state_dict": model.state_dict(),
    }

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, checkpoint_path)
    logger.debug(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """
    Load model checkpoint.

    Convenience function for simple checkpoint loading.

    Args:
        checkpoint_path: Path to checkpoint
        model: PyTorch model to load into
        optimizer: Optional optimizer to load state into

    Returns:
        Round number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    round_num = checkpoint.get("round", 0)

    logger.info(f"Loaded checkpoint from round {round_num}")

    return round_num


def save_model_for_serving(
    model: torch.nn.Module,
    save_path: Path,
    sample_input: Optional[torch.Tensor] = None,
) -> None:
    """
    Save model in format suitable for serving.

    Args:
        model: PyTorch model
        save_path: Path to save model
        sample_input: Sample input for tracing (for TorchScript)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save state dict
    state_dict_path = save_path.with_suffix(".pt")
    torch.save(model.state_dict(), state_dict_path)

    logger.info(f"Saved model state dict to {state_dict_path}")

    # Try to save TorchScript if sample input provided
    if sample_input is not None:
        try:
            model.eval()
            traced_model = torch.jit.trace(model, sample_input)
            torchscript_path = save_path.with_suffix(".torchscript")
            traced_model.save(torchscript_path)
            logger.info(f"Saved TorchScript model to {torchscript_path}")
        except Exception as e:
            logger.warning(f"Failed to create TorchScript model: {e}")


def load_model_for_serving(
    model: torch.nn.Module,
    load_path: Path,
) -> torch.nn.Module:
    """
    Load model for serving.

    Args:
        model: Model instance (architecture)
        load_path: Path to saved model

    Returns:
        Loaded model
    """
    state_dict = torch.load(load_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded model from {load_path}")

    return model


def export_to_onnx(
    model: torch.nn.Module,
    save_path: Path,
    sample_input: torch.Tensor,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
) -> None:
    """
    Export model to ONNX format.

    Args:
        model: PyTorch model
        save_path: Path to save ONNX model
        sample_input: Sample input for tracing
        input_names: Input tensor names
        output_names: Output tensor names
    """
    model.eval()

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    torch.onnx.export(
        model,
        sample_input,
        save_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=14,
    )

    logger.info(f"Exported model to ONNX: {save_path}")
