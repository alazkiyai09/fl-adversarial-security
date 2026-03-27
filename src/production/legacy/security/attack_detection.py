"""Attack detection for federated learning.

Detects various types of attacks:
- Data poisoning (malicious training data)
- Model poisoning (malicious model updates)
- Backdoor attacks (hidden triggers)
- Label flipping (changing labels)
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

import torch
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from loguru import logger

from ..utils.logging import get_fl_logger


class AttackType(Enum):
    """Types of attacks."""
    POISONING = "poisoning"
    BACKDOOR = "backdoor"
    LABEL_FLIPPING = "label_flipping"
    BYZANTINE = "byzantine"
    OMNISCIENT = "omniscient"


@dataclass
class AttackResult:
    """Result of attack detection."""
    is_malicious: bool
    confidence: float
    attack_type: Optional[AttackType]
    details: Dict[str, Any]


class PoisoningDetector:
    """
    Detects data/model poisoning attacks.

    Methods:
    - Update magnitude analysis
    - Direction similarity analysis
    - Clustering-based detection
    - Statistical outlier detection
    """

    def __init__(
        self,
        threshold: float = 3.0,
        method: str = "statistical",
        window_size: int = 10,
    ):
        """
        Initialize poisoning detector.

        Args:
            threshold: Z-score threshold for outlier detection
            method: Detection method ('statistical', 'clustering', 'isolation_forest', 'hybrid')
            window_size: Window for historical analysis
        """
        self.threshold = threshold
        self.method = method
        self.window_size = window_size

        # Historical data
        self.update_magnitudes: List[float] = []
        self.update_directions: List[np.ndarray] = []

        self.logger = get_fl_logger("poisoning_detector")

    def detect(
        self,
        updates: List[List[torch.Tensor]],
        baseline_metrics: Optional[Dict] = None,
    ) -> List[AttackResult]:
        """
        Detect poisoned updates.

        Args:
            updates: List of client updates (each is list of layer tensors)
            baseline_metrics: Baseline metrics for comparison

        Returns:
            List of AttackResult for each client
        """
        if len(updates) == 0:
            return []

        # Flatten updates
        flattened_updates = [self._flatten_update(update) for update in updates]

        # Compute features
        magnitudes = [np.linalg_norm(flat) for flat in flattened_updates]
        directions = [flat / (np.linalg_norm(flat) + 1e-8) for flat in flattened_updates]

        # Update history
        self.update_magnitudes.extend(magnitudes)
        if len(self.update_magnitudes) > self.window_size * len(updates):
            self.update_magnitudes = self.update_magnitudes[-self.window_size * len(updates):]

        self.update_directions.extend(directions)
        if len(self.update_directions) > self.window_size * len(updates):
            self.update_directions = self.update_directions[-self.window_size * len(updates):]

        # Detect based on method
        if self.method == "statistical":
            results = self._statistical_detection(flattened_updates, magnitudes, directions)
        elif self.method == "clustering":
            results = self._clustering_detection(flattened_updates)
        elif self.method == "isolation_forest":
            results = self._isolation_forest_detection(flattened_updates)
        elif self.method == "hybrid":
            results = self._hybrid_detection(flattened_updates, magnitudes, directions)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

        # Log detection results
        n_malicious = sum(1 for r in results if r.is_malicious)
        self.logger.info(
            f"Poisoning detection: {n_malicious}/{len(results)} malicious clients detected "
            f"(method: {self.method})"
        )

        return results

    def _flatten_update(self, update: List[torch.Tensor]) -> np.ndarray:
        """Flatten update to 1D array."""
        flattened = torch.cat([layer.flatten() for layer in update])
        return flattened.cpu().numpy()

    def _statistical_detection(
        self,
        updates: List[np.ndarray],
        magnitudes: List[float],
        directions: List[np.ndarray],
    ) -> List[AttackResult]:
        """Detect using statistical outlier detection."""
        results = []

        # Z-score on magnitudes
        if len(self.update_magnitudes) > 2:
            z_scores = np.abs(stats.zscore(magnitudes))
        else:
            z_scores = np.zeros(len(magnitudes))

        # Direction similarity to others
        direction_scores = []
        for i, dir_i in enumerate(directions):
            similarities = []
            for j, dir_j in enumerate(directions):
                if i != j:
                    # Cosine similarity
                    sim = np.dot(dir_i, dir_j)
                    similarities.append(sim)
            direction_scores.append(np.mean(similarities) if similarities else 0.0)

        # Combine scores
        for i in range(len(updates)):
            # Flag as malicious if:
            # 1. Magnitude is outlier (high z-score)
            # 2. Direction is very different from others
            magnitude_outlier = z_scores[i] > self.threshold
            direction_outlier = direction_scores[i] < -0.5  # Very different

            is_malicious = magnitude_outlier or direction_outlier

            # Confidence based on combined evidence
            confidence = min(1.0, (z_scores[i] / self.threshold) if magnitude_outlier else 0.5)
            if direction_outlier:
                confidence = max(confidence, 0.7)

            results.append(AttackResult(
                is_malicious=is_malicious,
                confidence=float(confidence),
                attack_type=AttackType.POISONING if is_malicious else None,
                details={
                    "magnitude_z_score": float(z_scores[i]),
                    "direction_similarity": float(direction_scores[i]),
                    "magnitude_outlier": magnitude_outlier,
                    "direction_outlier": direction_outlier,
                },
            ))

        return results

    def _clustering_detection(self, updates: List[np.ndarray]) -> List[AttackResult]:
        """Detect using clustering (DBSCAN)."""
        try:
            # Normalize updates
            normalized = np.array([u / (np.linalg.norm(u) + 1e-8) for u in updates])

            # Apply DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=max(2, len(updates) // 3))
            labels = clustering.fit_predict(normalized)

            # Identify outliers (label = -1)
            results = []
            for i, label in enumerate(labels):
                is_malicious = label == -1
                confidence = 0.8 if is_malicious else 0.1

                results.append(AttackResult(
                    is_malicious=is_malicious,
                    confidence=confidence,
                    attack_type=AttackType.POISONING if is_malicious else None,
                    details={"cluster_label": int(label)},
                ))

            return results

        except Exception as e:
            self.logger.warning(f"Clustering detection failed: {e}")
            # Return all benign if clustering fails
            return [AttackResult(
                is_malicious=False,
                confidence=0.0,
                attack_type=None,
                details={},
            ) for _ in updates]

    def _isolation_forest_detection(self, updates: List[np.ndarray]) -> List[AttackResult]:
        """Detect using Isolation Forest."""
        try:
            # Normalize updates
            normalized = np.array([u / (np.linalg.norm(u) + 1e-8) for u in updates])

            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(normalized)
            scores = iso_forest.score_samples(normalized)

            # Convert predictions (1 = benign, -1 = malicious)
            results = []
            for i, pred in enumerate(predictions):
                is_malicious = pred == -1
                # Anomaly score is negative, convert to [0, 1] confidence
                confidence = float(max(0.0, -scores[i]))

                results.append(AttackResult(
                    is_malicious=is_malicious,
                    confidence=confidence,
                    attack_type=AttackType.POISONING if is_malicious else None,
                    details={"anomaly_score": float(scores[i])},
                ))

            return results

        except Exception as e:
            self.logger.warning(f"Isolation Forest detection failed: {e}")
            return [AttackResult(
                is_malicious=False,
                confidence=0.0,
                attack_type=None,
                details={},
            ) for _ in updates]

    def _hybrid_detection(
        self,
        updates: List[np.ndarray],
        magnitudes: List[float],
        directions: List[np.ndarray],
    ) -> List[AttackResult]:
        """Combine multiple detection methods."""
        # Get results from all methods
        stat_results = self._statistical_detection(updates, magnitudes, directions)
        cluster_results = self._clustering_detection(updates)
        iso_results = self._isolation_forest_detection(updates)

        # Combine using voting
        results = []
        for i in range(len(updates)):
            votes = sum([
                stat_results[i].is_malicious,
                cluster_results[i].is_malicious,
                iso_results[i].is_malicious,
            ])

            # Require majority vote
            is_malicious = votes >= 2

            # Average confidence
            confidence = (
                stat_results[i].confidence +
                cluster_results[i].confidence +
                iso_results[i].confidence
            ) / 3.0

            results.append(AttackResult(
                is_malicious=is_malicious,
                confidence=float(confidence),
                attack_type=AttackType.POISONING if is_malicious else None,
                details={
                    "votes": int(votes),
                    "statistical": stat_results[i].is_malicious,
                    "clustering": cluster_results[i].is_malicious,
                    "isolation_forest": iso_results[i].is_malicious,
                },
            ))

        return results


class BackdoorDetector:
    """
    Detects backdoor attacks.

    Methods:
    - Model behavior on backdoor test set
    - Neuron activation analysis
    - Gradient analysis
    """

    def __init__(
        self,
        backdoor_test_set: Optional[torch.utils.data.DataLoader] = None,
        threshold: float = 0.9,
    ):
        """
        Initialize backdoor detector.

        Args:
            backdoor_test_set: Test set with potential backdoor triggers
            threshold: Prediction rate threshold for backdoor detection
        """
        self.backdoor_test_set = backdoor_test_set
        self.threshold = threshold

        self.logger = get_fl_logger("backdoor_detector")

    def detect(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> AttackResult:
        """
        Detect backdoor in model.

        Args:
            model: Model to test
            device: Device for inference

        Returns:
            AttackResult
        """
        if self.backdoor_test_set is None:
            return AttackResult(
                is_malicious=False,
                confidence=0.0,
                attack_type=None,
                details={"message": "No backdoor test set provided"},
            )

        model.eval()

        # Test on backdoor test set
        backdoor_target_count = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in self.backdoor_test_set:
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1).cpu()

                # Check if predictions are consistently the same class
                # (potential backdoor target)
                if len(pred) > 0:
                    most_common = torch.mode(pred).values.item()
                    backdoor_target_count += (pred == most_common).sum().item()
                    total_samples += len(pred)

        if total_samples == 0:
            return AttackResult(
                is_malicious=False,
                confidence=0.0,
                attack_type=None,
                details={"message": "Empty backdoor test set"},
            )

        # High consistency suggests backdoor
        consistency_rate = backdoor_target_count / total_samples
        is_malicious = consistency_rate > self.threshold
        confidence = float(consistency_rate)

        self.logger.info(
            f"Backdoor detection: consistency rate = {consistency_rate:.4f}, "
            f"malicious = {is_malicious}"
        )

        return AttackResult(
            is_malicious=is_malicious,
            confidence=confidence,
            attack_type=AttackType.BACKDOOR if is_malicious else None,
            details={
                "consistency_rate": float(consistency_rate),
                "backdoor_target_count": backdoor_target_count,
                "total_samples": total_samples,
            },
        )

    def check_trigger_effect(
        self,
        model: torch.nn.Module,
        clean_data: torch.Tensor,
        triggered_data: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Check effect of trigger on model predictions.

        Args:
            model: Model to test
            clean_data: Clean input data
            triggered_data: Data with backdoor trigger
            device: Device for inference

        Returns:
            Dictionary of metrics
        """
        model.eval()

        with torch.no_grad():
            clean_output = model(clean_data.to(device))
            triggered_output = model(triggered_data.to(device))

        clean_pred = clean_output.argmax(dim=1)
        triggered_pred = triggered_output.argmax(dim=1)

        # Compute change in predictions
        prediction_change_rate = (clean_pred != triggered_pred).float().mean().item()

        # Compute output confidence change
        clean_confidence = torch.softmax(clean_output, dim=1).max(dim=1).values.mean().item()
        triggered_confidence = torch.softmax(triggered_output, dim=1).max(dim=1).values.mean().item()

        return {
            "prediction_change_rate": prediction_change_rate,
            "clean_confidence": clean_confidence,
            "triggered_confidence": triggered_confidence,
        }


class LabelFlippingDetector:
    """
    Detects label flipping attacks.

    Detects clients that flip labels (e.g., fraud → legitimate).
    """

    def __init__(
        self,
        expected_fraud_rate: float = 0.05,
        tolerance: float = 0.1,
    ):
        """
        Initialize label flipping detector.

        Args:
            expected_fraud_rate: Expected fraud rate in data
            tolerance: Tolerance for deviation
        """
        self.expected_fraud_rate = expected_fraud_rate
        self.tolerance = tolerance

        self.logger = get_fl_logger("label_flipping_detector")

    def detect(
        self,
        client_predictions: np.ndarray,
        client_labels: np.ndarray,
    ) -> AttackResult:
        """
        Detect label flipping in client's predictions.

        Args:
            client_predictions: Client's predictions
            client_labels: True labels

        Returns:
            AttackResult
        """
        # Compute fraud rate in predictions
        predicted_fraud_rate = client_predictions.mean()

        # Compute deviation from expected
        deviation = abs(predicted_fraud_rate - self.expected_fraud_rate)

        # Flag if deviation exceeds tolerance
        is_malicious = deviation > self.tolerance

        # Confidence based on deviation
        confidence = min(1.0, deviation / self.tolerance)

        # Check if labels are systematically flipped
        if is_malicious:
            # Compare prediction distribution to label distribution
            actual_fraud_rate = client_labels.mean()
            if actual_fraud_rate > 0:
                flip_ratio = abs(predicted_fraud_rate - actual_fraud_rate) / actual_fraud_rate
                confidence = max(confidence, flip_ratio)

        self.logger.info(
            f"Label flipping detection: predicted fraud rate = {predicted_fraud_rate:.4f}, "
            f"deviation = {deviation:.4f}, malicious = {is_malicious}"
        )

        return AttackResult(
            is_malicious=is_malicious,
            confidence=float(confidence),
            attack_type=AttackType.LABEL_FLIPPING if is_malicious else None,
            details={
                "predicted_fraud_rate": float(predicted_fraud_rate),
                "deviation": float(deviation),
                "expected_fraud_rate": self.expected_fraud_rate,
            },
        )


class AttackDetector:
    """
    Main attack detection orchestrator.

    Combines multiple detection methods.
    """

    def __init__(self, config: Dict):
        """
        Initialize attack detector.

        Args:
            config: Detection configuration
        """
        self.config = config

        # Initialize detectors
        self.poisoning_detector = PoisoningDetector(
            threshold=config.get("anomaly_threshold", 3.0),
            method=config.get("method", "statistical"),
        )

        self.backdoor_detector = None  # Requires backdoor test set
        self.label_flipping_detector = LabelFlippingDetector(
            expected_fraud_rate=config.get("expected_fraud_rate", 0.05),
            tolerance=config.get("tolerance", 0.1),
        )

        self.logger = get_fl_logger("attack_detector")

    def detect_poisoning(
        self,
        updates: List[List[torch.Tensor]],
        baseline_metrics: Optional[Dict] = None,
    ) -> List[AttackResult]:
        """
        Detect poisoning attacks.

        Args:
            updates: Client updates
            baseline_metrics: Baseline metrics

        Returns:
            List of AttackResult
        """
        return self.poisoning_detector.detect(updates, baseline_metrics)

    def detect_backdoor(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> AttackResult:
        """
        Detect backdoor attacks.

        Args:
            model: Model to test
            device: Device for inference

        Returns:
            AttackResult
        """
        if self.backdoor_detector is None:
            self.backdoor_detector = BackdoorDetector()

        return self.backdoor_detector.detect(model, device)

    def detect_all_attacks(
        self,
        updates: List[List[torch.Tensor]],
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Run all attack detection methods.

        Args:
            updates: Client updates
            model: Optional model for backdoor detection
            device: Optional device for inference

        Returns:
            Dictionary of detection results
        """
        results = {
            "poisoning": self.detect_poisoning(updates),
        }

        if model is not None and device is not None:
            results["backdoor"] = self.detect_backdoor(model, device)

        # Summary
        n_poisoning = sum(1 for r in results["poisoning"] if r.is_malicious)
        n_backdoor = 1 if "backdoor" in results and results["backdoor"].is_malicious else 0

        results["summary"] = {
            "n_malicious_poisoning": n_poisoning,
            "n_backdoor_detected": n_backdoor,
            "total_clients": len(updates),
        }

        self.logger.info(
            f"Attack detection summary: {n_poisoning}/{len(updates)} poisoning, "
            f"{n_backdoor} backdoor"
        )

        return results

    def set_backdoor_test_set(self, test_set: torch.utils.data.DataLoader) -> None:
        """
        Set backdoor test set.

        Args:
            test_set: Test set with potential triggers
        """
        self.backdoor_detector = BackdoorDetector(backdoor_test_set=test_set)
        self.logger.info("Backdoor test set configured")
