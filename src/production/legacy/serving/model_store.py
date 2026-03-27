"""Model store for versioning and managing ML models."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import shutil
import hashlib

from loguru import logger

from .prediction import ModelMetadata, Predictor


@dataclass
class ModelInfo:
    """Information about a stored model."""
    version: str
    path: Path
    model_type: str
    created_at: str
    is_active: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelStore:
    """
    Manages model versioning and storage.

    Features:
    - Version tracking
    - Model activation/rollback
    - Metadata management
    - Automatic cleanup of old versions
    """

    def __init__(
        self,
        store_path: Path,
        max_versions: int = 5,
    ):
        """
        Initialize model store.

        Args:
            store_path: Directory for storing models
            max_versions: Maximum number of versions to keep
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.max_versions = max_versions

        # Metadata file
        self.metadata_file = self.store_path / "models.json"

        # Load metadata
        self.models: Dict[str, ModelInfo] = {}
        self._load_metadata()

        # Active model
        self.active_version: Optional[str] = None
        self._load_active_version()

        logger.info(
            f"ModelStore initialized: {len(self.models)} versions, "
            f"active={self.active_version}"
        )

    def _load_metadata(self) -> None:
        """Load model metadata from file."""
        if not self.metadata_file.exists():
            return

        try:
            with open(self.metadata_file, "r") as f:
                data = json.load(f)

            for version_str, model_data in data.items():
                self.models[version_str] = ModelInfo(
                    version=model_data["version"],
                    path=Path(model_data["path"]),
                    model_type=model_data["model_type"],
                    created_at=model_data["created_at"],
                    is_active=model_data.get("is_active", False),
                    metadata=model_data.get("metadata", {}),
                )

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    def _save_metadata(self) -> None:
        """Save model metadata to file."""
        try:
            data = {}
            for version, info in self.models.items():
                data[version] = {
                    "version": info.version,
                    "path": str(info.path),
                    "model_type": info.model_type,
                    "created_at": info.created_at,
                    "is_active": info.is_active,
                    "metadata": info.metadata,
                }

            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _load_active_version(self) -> None:
        """Load active version from file."""
        active_file = self.store_path / "active.txt"

        if active_file.exists():
            try:
                with open(active_file, "r") as f:
                    self.active_version = f.read().strip()
            except Exception as e:
                logger.error(f"Failed to load active version: {e}")

    def _save_active_version(self) -> None:
        """Save active version to file."""
        active_file = self.store_path / "active.txt"

        try:
            with open(active_file, "w") as f:
                f.write(self.active_version or "")
        except Exception as e:
            logger.error(f"Failed to save active version: {e}")

    def save_model(
        self,
        model,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a model to the store.

        Args:
            model: PyTorch model or Predictor
            version: Version string (auto-generated if None)
            metadata: Additional metadata
            config: Model configuration

        Returns:
            Version string
        """
        # Generate version if not provided
        if version is None:
            version = self._generate_version()

        # Create version directory
        version_path = self.store_path / version
        version_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = version_path / "model.pt"

        try:
            import torch

            if isinstance(model, Predictor):
                # Save Predictor
                torch.save({
                    "model_state_dict": model.model.state_dict(),
                    "metadata": {
                        "version": model.metadata.version,
                        "model_type": model.metadata.model_type,
                        "training_round": model.metadata.training_round,
                    },
                    "config": model.config,
                }, model_path)

                model_type = model.metadata.model_type

            else:
                # Save raw PyTorch model
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config or {},
                }, model_path)

                model_type = config.get("model_type", "unknown") if config else "unknown"

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

        # Create metadata
        model_info = ModelInfo(
            version=version,
            path=model_path,
            model_type=model_type,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        self.models[version] = model_info
        self._save_metadata()

        # Cleanup old versions
        self._cleanup_old_versions()

        logger.info(f"Saved model version {version} to {model_path}")

        return version

    def load_model(
        self,
        version: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Any] = None,
    ) -> Predictor:
        """
        Load a model from the store.

        Args:
            version: Version to load (active if None)
            config: Model configuration
            device: Device for inference

        Returns:
            Predictor instance
        """
        # Get version
        if version is None:
            version = self.active_version

        if version is None:
            raise ValueError("No active model version")

        if version not in self.models:
            raise ValueError(f"Model version {version} not found")

        model_info = self.models[version]

        # Load predictor from checkpoint
        predictor = Predictor.from_checkpoint(
            checkpoint_path=model_info.path,
            config=config or {},
            device=device,
        )

        logger.info(f"Loaded model version {version}")

        return predictor

    def activate_model(self, version: str) -> None:
        """
        Activate a model version.

        Args:
            version: Version to activate
        """
        if version not in self.models:
            raise ValueError(f"Model version {version} not found")

        # Deactivate current active
        if self.active_version and self.active_version in self.models:
            self.models[self.active_version].is_active = False

        # Activate new version
        self.models[version].is_active = True
        self.active_version = version

        self._save_metadata()
        self._save_active_version()

        logger.info(f"Activated model version {version}")

    def get_active_version(self) -> Optional[str]:
        """Get active model version."""
        return self.active_version

    def get_active_predictor(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Any] = None,
    ) -> Optional[Predictor]:
        """
        Get predictor for active model.

        Args:
            config: Model configuration
            device: Device for inference

        Returns:
            Predictor instance or None if no active model
        """
        if self.active_version is None:
            return None

        return self.load_model(self.active_version, config, device)

    def rollback(self, target_version: Optional[str] = None) -> bool:
        """
        Rollback to a previous version.

        Args:
            target_version: Specific version to rollback to (previous if None)

        Returns:
            True if rollback successful
        """
        if target_version is None:
            # Rollback to previous version
            versions = sorted(self.models.keys(), reverse=True)

            if len(versions) < 2:
                logger.warning("No previous version to rollback to")
                return False

            # Find previous non-active version
            for v in versions:
                if v != self.active_version:
                    target_version = v
                    break

        if target_version not in self.models:
            logger.error(f"Target version {target_version} not found")
            return False

        try:
            self.activate_model(target_version)
            logger.info(f"Rolled back to version {target_version}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all model versions.

        Returns:
            List of version information dictionaries
        """
        versions = []

        for version, info in sorted(self.models.items()):
            versions.append({
                "version": version,
                "model_type": info.model_type,
                "created_at": info.created_at,
                "is_active": info.is_active,
                "metadata": info.metadata,
            })

        return versions

    def get_model_info(self, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about a model version.

        Args:
            version: Version to query (active if None)

        Returns:
            Model information dictionary
        """
        if version is None:
            version = self.active_version

        if version not in self.models:
            return None

        info = self.models[version]

        return {
            "version": info.version,
            "model_type": info.model_type,
            "created_at": info.created_at,
            "is_active": info.is_active,
            "path": str(info.path),
            "metadata": info.metadata,
        }

    def delete_version(self, version: str) -> bool:
        """
        Delete a model version.

        Args:
            version: Version to delete

        Returns:
            True if deletion successful
        """
        if version not in self.models:
            logger.warning(f"Version {version} not found")
            return False

        # Don't delete active model
        if version == self.active_version:
            logger.error("Cannot delete active model version")
            return False

        info = self.models[version]

        try:
            # Delete model file
            if info.path.exists():
                info.path.unlink()

            # Delete version directory if empty
            version_dir = info.path.parent
            if version_dir.exists() and not list(version_dir.iterdir()):
                version_dir.rmdir()

            # Remove from metadata
            del self.models[version]
            self._save_metadata()

            logger.info(f"Deleted model version {version}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete version {version}: {e}")
            return False

    def _cleanup_old_versions(self) -> None:
        """Remove old versions exceeding max_versions."""
        versions = sorted(self.models.keys(), reverse=True)

        if len(versions) <= self.max_versions:
            return

        # Keep active version and max_versions - 1 most recent
        versions_to_keep = set()

        # Always keep active
        if self.active_version:
            versions_to_keep.add(self.active_version)

        # Keep most recent
        for v in versions:
            if len(versions_to_keep) >= self.max_versions:
                break
            versions_to_keep.add(v)

        # Delete old versions
        for v in versions:
            if v not in versions_to_keep:
                self.delete_version(v)

    def _generate_version(self) -> str:
        """Generate a unique version string."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(timestamp.encode()).hexdigest()[:6]
        return f"v_{timestamp}_{hash_suffix}"

    def export_model(
        self,
        version: str,
        output_path: Path,
        format: str = "torchscript",
    ) -> bool:
        """
        Export a model version.

        Args:
            version: Version to export
            output_path: Output path
            format: Export format ('torchscript', 'onnx')

        Returns:
            True if export successful
        """
        if version not in self.models:
            logger.error(f"Version {version} not found")
            return False

        try:
            import torch

            # Load model
            checkpoint = torch.load(self.models[version].path, map_location="cpu")
            model_state_dict = checkpoint["model_state_dict"]

            # Recreate model (simplified)
            from ..models import LSTMFraudDetector
            model = LSTMFraudDetector(
                input_size=30,  # Would come from config
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
            )
            model.load_state_dict(model_state_dict)
            model.eval()

            output_path = Path(output_path)

            if format == "torchscript":
                # Export as TorchScript
                dummy_input = torch.randn(1, 10, 30)
                traced_model = torch.jit.trace(model, dummy_input)
                traced_model.save(output_path)

            elif format == "onnx":
                # Export as ONNX
                dummy_input = torch.randn(1, 10, 30)
                torch.onnx.export(
                    model,
                    dummy_input,
                    output_path,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )

            else:
                logger.error(f"Unknown export format: {format}")
                return False

            logger.info(f"Exported version {version} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
