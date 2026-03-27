"""Base class for anomaly detectors."""

from abc import ABC, abstractmethod
from typing import List
from src.defenses.signguard_full.legacy.core.types import ModelUpdate
import torch


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @abstractmethod
    def compute_score(
        self,
        update: ModelUpdate,
        global_model: dict[str, torch.Tensor],
        client_history: List[ModelUpdate] | None = None,
    ) -> float:
        """Compute anomaly score for a single update."""
        pass
