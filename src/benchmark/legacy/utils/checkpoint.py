"""
Checkpoint management for saving and loading experiment state.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json


class CheckpointManager:
    """
    Manager for saving and loading experiment checkpoints.
    """

    def __init__(self, checkpoint_dir: str = "results/checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        checkpoint_name: str,
    ) -> Path:
        """
        Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint dictionary containing model state and metadata
            checkpoint_name: Name for the checkpoint file

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"

        # Prepare checkpoint
        save_dict = {}
        for key, value in checkpoint.items():
            if isinstance(value, torch.nn.Module):
                save_dict[f"{key}_state_dict"] = value.state_dict()
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                save_dict[key] = value
            else:
                save_dict[key] = value

        torch.save(save_dict, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """
        Load checkpoint from disk.

        Args:
            checkpoint_name: Name of the checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return torch.load(checkpoint_path)

    def save_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save model with metadata.

        Args:
            model: PyTorch model to save
            model_name: Name for the model file
            metadata: Optional metadata dictionary

        Returns:
            Path to saved model
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }
        return self.save_checkpoint(checkpoint, model_name)

    def load_model(
        self,
        model: torch.nn.Module,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Load model state into provided model.

        Args:
            model: PyTorch model to load state into
            model_name: Name of the model file

        Returns:
            Metadata dictionary
        """
        checkpoint = self.load_checkpoint(model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint.get("metadata", {})

    def save_results(
        self,
        results: Dict[str, Any],
        results_name: str,
    ) -> Path:
        """
        Save results as JSON.

        Args:
            results: Results dictionary
            results_name: Name for the results file

        Returns:
            Path to saved results
        """
        results_path = self.checkpoint_dir / f"{results_name}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.integer):
                serializable_results[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        return results_path

    def load_results(self, results_name: str) -> Dict[str, Any]:
        """
        Load results from JSON.

        Args:
            results_name: Name of the results file

        Returns:
            Results dictionary
        """
        results_path = self.checkpoint_dir / f"{results_name}.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Results not found: {results_path}")

        with open(results_path, "r") as f:
            return json.load(f)

    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint names
        """
        return [f.stem for f in self.checkpoint_dir.glob("*.pt")]

    def delete_checkpoint(self, checkpoint_name: str) -> None:
        """
        Delete checkpoint.

        Args:
            checkpoint_name: Name of checkpoint to delete
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
