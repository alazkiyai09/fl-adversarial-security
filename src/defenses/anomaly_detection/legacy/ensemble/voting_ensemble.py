"""
Voting ensemble for combining multiple anomaly detectors.
Improves detection by leveraging complementary methods.
"""

from typing import List, Dict, Literal
import numpy as np

from src.defenses.anomaly_detection.legacy.detectors.base_detector import BaseDetector


class VotingEnsemble:
    """
    Combine multiple detectors using voting strategies.

    Strategies:
    - majority: Flag if >= 50% of detectors flag as malicious
    - unanimous: Flag if ALL detectors flag as malicious (conservative)
    - weighted: Weighted average of scores (configurable weights)
    - soft: Average of all anomaly scores (soft voting)

    Benefits:
    - Reduces false positives (requires consensus)
    - Handles diverse attack types (different detectors catch different attacks)
    - More robust than any single detector
    """

    def __init__(
        self,
        detectors: List[BaseDetector],
        voting: Literal["majority", "unanimous", "weighted", "soft"] = "majority",
        weights: Dict[str, float] = None
    ):
        """
        Initialize voting ensemble.

        Args:
            detectors: List of fitted detector instances
            voting: Voting strategy
            weights: Detector weights (for "weighted" voting)
                     Maps detector class name -> weight
        """
        if not detectors:
            raise ValueError("detectors list cannot be empty")

        self.detectors = detectors
        self.voting = voting
        self.weights = weights or {}

        # Build default weights (equal for all)
        if not self.weights:
            for detector in detectors:
                self.weights[detector.__class__.__name__] = 1.0

    def compute_anomaly_score(self, update, **kwargs) -> float:
        """
        Compute ensemble anomaly score.

        Args:
            update: Model update (format depends on detectors)
            **kwargs: Additional parameters (e.g., global_model, client_id)

        Returns:
            Ensemble anomaly score
        """
        scores = {}
        for detector in self.detectors:
            name = detector.__class__.__name__
            try:
                score = detector.compute_anomaly_score(update, **kwargs)
                scores[name] = score
            except Exception as e:
                # If detector fails, skip it
                print(f"Warning: {name} failed: {e}")
                scores[name] = 0.0

        if self.voting == "soft":
            # Soft voting: average of all scores
            return float(np.mean(list(scores.values())))

        elif self.voting == "weighted":
            # Weighted average of scores
            weighted_sum = 0.0
            total_weight = 0.0
            for name, score in scores.items():
                weight = self.weights.get(name, 1.0)
                weighted_sum += weight * score
                total_weight += weight

            return weighted_sum / total_weight if total_weight > 0 else 0.0

        else:
            # For majority/unanimous, return max score
            return float(np.max(list(scores.values())))

    def is_malicious(self, update, **kwargs) -> bool:
        """
        Binary decision: is client malicious?

        Args:
            update: Model update
            **kwargs: Additional parameters

        Returns:
            True if ensemble flags as malicious
        """
        votes = {}
        for detector in self.detectors:
            name = detector.__class__.__name__
            try:
                vote = detector.is_malicious(update, **kwargs)
                votes[name] = vote
            except Exception as e:
                # If detector fails, treat as not malicious (conservative)
                print(f"Warning: {name} failed: {e}")
                votes[name] = False

        if self.voting == "majority":
            # Majority vote: > 50% must flag
            num_flags = sum(votes.values())
            return num_flags > len(self.detectors) / 2

        elif self.voting == "unanimous":
            # Unanimous: ALL must flag
            return all(votes.values())

        elif self.voting == "soft":
            # Soft voting: compare average score to threshold
            score = self.compute_anomaly_score(update, **kwargs)
            # Use mean threshold of all detectors
            avg_threshold = np.mean([d.threshold for d in self.detectors])
            return score > avg_threshold

        elif self.voting == "weighted":
            # Weighted voting
            score = self.compute_anomaly_score(update, **kwargs)
            # Weighted threshold
            weighted_threshold = sum(
                self.weights.get(d.__class__.__name__, 1.0) * d.threshold
                for d in self.detectors
            ) / sum(self.weights.values())
            return score > weighted_threshold

        else:
            raise ValueError(f"Unknown voting strategy: {self.voting}")

    def get_individual_scores(self, update, **kwargs) -> Dict[str, float]:
        """
        Get individual detector scores (for analysis).

        Args:
            update: Model update
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping detector name -> anomaly score
        """
        scores = {}
        for detector in self.detectors:
            name = detector.__class__.__name__
            try:
                score = detector.compute_anomaly_score(update, **kwargs)
                scores[name] = score
            except Exception as e:
                scores[name] = -1.0  # Error indicator

        return scores

    def get_voting_summary(self, update, **kwargs) -> Dict[str, any]:
        """
        Get detailed voting summary (for analysis).

        Args:
            update: Model update
            **kwargs: Additional parameters

        Returns:
            Dictionary with voting details
        """
        votes = {}
        scores = {}

        for detector in self.detectors:
            name = detector.__class__.__name__
            try:
                votes[name] = detector.is_malicious(update, **kwargs)
                scores[name] = detector.compute_anomaly_score(update, **kwargs)
            except Exception as e:
                votes[name] = False
                scores[name] = -1.0

        num_flags = sum(votes.values())

        return {
            'votes': votes,
            'scores': scores,
            'num_flags': num_flags,
            'num_detectors': len(self.detectors),
            'flag_percentage': num_flags / len(self.detectors),
            'ensemble_decision': self.is_malicious(update, **kwargs),
            'ensemble_score': self.compute_anomaly_score(update, **kwargs)
        }
