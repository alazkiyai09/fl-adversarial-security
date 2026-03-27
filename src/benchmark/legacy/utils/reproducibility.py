"""
Reproducibility utilities for setting random seeds and managing randomness.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed() -> int:
    """
    Get a random seed value.

    Returns:
        Random integer seed
    """
    return random.randint(0, 2**32 - 1)


class SeedManager:
    """
    Context manager for temporarily setting seeds.
    """

    def __init__(self, seed: int, deterministic: bool = True):
        """
        Initialize seed manager.

        Args:
            seed: Random seed value
            deterministic: If True, enable deterministic algorithms
        """
        self.seed = seed
        self.deterministic = deterministic
        self._prev_state: Optional[dict] = None

    def __enter__(self):
        """Save current state and set new seed."""
        self._prev_state = {
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(),
        }
        set_seed(self.seed, self.deterministic)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous random state."""
        if self._prev_state is not None:
            random.setstate(self._prev_state["random_state"])
            np.random.set_state(self._prev_state["numpy_state"])
            torch.set_rng_state(self._prev_state["torch_state"])
        return False


class ReproducibilityInfo:
    """
    Container for reproducibility information.
    """

    def __init__(
        self,
        seed: int,
        torch_version: str,
        cuda_version: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize reproducibility info.

        Args:
            seed: Random seed used
            torch_version: PyTorch version
            cuda_version: CUDA version if available
            device: Device used for computation
        """
        self.seed = seed
        self.torch_version = torch_version
        self.cuda_version = cuda_version
        self.device = device

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "seed": self.seed,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "device": self.device,
        }

    @staticmethod
    def capture(seed: int) -> "ReproducibilityInfo":
        """
        Capture current environment info.

        Args:
            seed: Random seed being used

        Returns:
            ReproducibilityInfo instance
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None

        return ReproducibilityInfo(
            seed=seed,
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            device=device,
        )
