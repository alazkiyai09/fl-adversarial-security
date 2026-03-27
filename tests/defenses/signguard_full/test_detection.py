"""Tests for anomaly detection modules."""

import pytest
import torch
import numpy as np

from signguard.detection import (
    AnomalyDetector,
    L2NormDetector,
    CosineSimilarityDetector,
    LossDeviationDetector,
    EnsembleDetector,
)
from signguard.core.types import ModelUpdate


class TestL2NormDetector:
    """Tests for L2NormDetector."""

    @pytest.fixture
    def detector(self):
        """Create L2 norm detector."""
        return L2NormDetector(use_mad=True, window_size=10)

    @pytest.fixture
    def global_params(self):
        """Create global model parameters."""
        return {
            "layer1.weight": torch.randn(128, 28),
            "layer1.bias": torch.randn(128),
        }

    @pytest.fixture
    def normal_update(self, global_params):
        """Create normal update."""
        return ModelUpdate(
            client_id="client_0",
            round_num=1,
            parameters={
                name: param + torch.randn_like(param) * 0.01
                for name, param in global_params.items()
            },
            num_samples=100,
            metrics={"loss": 0.5},
        )

    @pytest.fixture
    def anomalous_update(self, global_params):
        """Create anomalous update (large magnitude)."""
        return ModelUpdate(
            client_id="client_0",
            round_num=1,
            parameters={
                name: param + torch.randn_like(param) * 10.0
                for name, param in global_params.items()
            },
            num_samples=100,
            metrics={"loss": 0.5},
        )

    def test_compute_score_normal_update(self, detector, normal_update, global_params):
        """Test score for normal update."""
        score = detector.compute_score(normal_update, global_params)
        assert 0.0 <= score <= 1.0
        # Normal update should have low anomaly score
        assert score < 0.5

    def test_compute_score_anomalous_update(self, detector, anomalous_update, global_params):
        """Test score for anomalous update."""
        score = detector.compute_score(anomalous_update, global_params)
        assert 0.0 <= score <= 1.0
        # Anomalous update should have higher score
        assert score > 0.3

    def test_update_statistics(self, detector, normal_update, global_params):
        """Test updating statistics."""
        updates = [normal_update] * 5
        detector.update_statistics(updates, global_params)
        assert len(detector.norm_history) == 5

    def test_window_size_limit(self, detector, normal_update, global_params):
        """Test window size limit."""
        updates = [normal_update] * 15
        detector.update_statistics(updates, global_params)
        assert len(detector.norm_history) == 10  # Limited by window_size

    def test_get_statistics(self, detector, normal_update, global_params):
        """Test getting statistics."""
        detector.update_statistics([normal_update] * 5, global_params)
        stats = detector.get_statistics()
        assert stats["count"] == 5
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats


class TestCosineSimilarityDetector:
    """Tests for CosineSimilarityDetector."""

    @pytest.fixture
    def detector(self):
        """Create cosine similarity detector."""
        return CosineSimilarityDetector(window_size=10)

    @pytest.fixture
    def global_params(self):
        """Create global model parameters."""
        return {
            "layer1.weight": torch.randn(128, 28),
            "layer1.bias": torch.randn(128),
        }

    @pytest.fixture
    def consistent_updates(self, global_params):
        """Create updates in consistent direction."""
        direction = {
            name: torch.randn_like(param) * 0.01
            for name, param in global_params.items()
        }
        return [
            ModelUpdate(
                client_id=f"client_{i}",
                round_num=1,
                parameters={
                    name: param + direction[name] * (i + 1) / 10.0
                    for name, param in global_params.items()
                },
                num_samples=100,
                metrics={"loss": 0.5},
            )
            for i in range(5)
        ]

    @pytest.fixture
    def opposite_update(self, global_params):
        """Create update in opposite direction."""
        return ModelUpdate(
            client_id="client_malicious",
            round_num=1,
            parameters={
                name: param - torch.randn_like(param) * 5.0
                for name, param in global_params.items()
            },
            num_samples=100,
            metrics={"loss": 0.5},
        )

    def test_compute_score_no_history(self, detector, global_params):
        """Test score when no history."""
        update = ModelUpdate(
            client_id="client_0",
            round_num=1,
            parameters={
                name: param + torch.randn_like(param) * 0.01
                for name, param in global_params.items()
            },
            num_samples=100,
        )
        score = detector.compute_score(update, global_params)
        # No history, should return 0
        assert score == 0.0

    def test_compute_score_with_history(self, detector, consistent_updates, global_params):
        """Test score with history."""
        detector.update_statistics(consistent_updates, global_params)
        
        # Consistent update should have low anomaly score
        new_update = ModelUpdate(
            client_id="client_new",
            round_num=2,
            parameters={
                name: param + torch.randn_like(param) * 0.01
                for name, param in global_params.items()
            },
            num_samples=100,
        )
        score = detector.compute_score(new_update, global_params)
        assert 0.0 <= score <= 1.0

    def test_update_statistics(self, detector, consistent_updates, global_params):
        """Test updating statistics."""
        detector.update_statistics(consistent_updates, global_params)
        assert len(detector.direction_history) == 5

    def test_get_statistics(self, detector, consistent_updates, global_params):
        """Test getting statistics."""
        detector.update_statistics(consistent_updates, global_params)
        stats = detector.get_statistics()
        assert stats["count"] == 5
        assert "mean_cosine_similarity" in stats


class TestLossDeviationDetector:
    """Tests for LossDeviationDetector."""

    @pytest.fixture
    def detector(self):
        """Create loss deviation detector."""
        return LossDeviationDetector(use_iqr=True, window_size=10)

    @pytest.fixture
    def global_params(self):
        """Create global model parameters."""
        return {"layer1.weight": torch.randn(128, 28)}

    @pytest.fixture
    def normal_updates(self, global_params):
        """Create updates with normal loss."""
        return [
            ModelUpdate(
                client_id=f"client_{i}",
                round_num=1,
                parameters=global_params,
                num_samples=100,
                metrics={"loss": 0.5 + np.random.randn() * 0.1},
            )
            for i in range(10)
        ]

    @pytest.fixture
    def high_loss_update(self, global_params):
        """Create update with abnormally high loss."""
        return ModelUpdate(
            client_id="client_malicious",
            round_num=1,
            parameters=global_params,
            num_samples=100,
            metrics={"loss": 10.0},
        )

    def test_compute_score_normal_loss(self, detector, normal_updates):
        """Test score for normal loss."""
        global_params = {"layer1.weight": torch.randn(128, 28)}
        detector.update_statistics(normal_updates, global_params)
        
        score = detector.compute_score(normal_updates[0], global_params)
        assert 0.0 <= score <= 1.0

    def test_compute_score_high_loss(self, detector, normal_updates, high_loss_update):
        """Test score for abnormally high loss."""
        global_params = {"layer1.weight": torch.randn(128, 28)}
        detector.update_statistics(normal_updates, global_params)
        
        score = detector.compute_score(high_loss_update, global_params)
        assert 0.0 <= score <= 1.0
        # High loss should produce higher anomaly score
        assert score > 0.3

    def test_compute_score_missing_loss(self, detector, global_params):
        """Test score when loss not in metrics."""
        update = ModelUpdate(
            client_id="client_0",
            round_num=1,
            parameters=global_params,
            num_samples=100,
            metrics={},  # No loss
        )
        score = detector.compute_score(update, global_params)
        # Missing loss should return 0 (assume normal)
        assert score == 0.0

    def test_update_statistics(self, detector, normal_updates):
        """Test updating statistics."""
        global_params = {"layer1.weight": torch.randn(128, 28)}
        detector.update_statistics(normal_updates, global_params)
        assert len(detector.loss_history) == 10

    def test_get_statistics(self, detector, normal_updates):
        """Test getting statistics."""
        global_params = {"layer1.weight": torch.randn(128, 28)}
        detector.update_statistics(normal_updates, global_params)
        stats = detector.get_statistics()
        assert stats["count"] == 10
        assert "mean" in stats
        assert "median" in stats
        assert "q25" in stats
        assert "q75" in stats


class TestEnsembleDetector:
    """Tests for EnsembleDetector."""

    @pytest.fixture
    def detector(self):
        """Create ensemble detector."""
        return EnsembleDetector(
            magnitude_weight=0.4,
            direction_weight=0.4,
            loss_weight=0.2,
            anomaly_threshold=0.7,
        )

    @pytest.fixture
    def global_params(self):
        """Create global model parameters."""
        return {
            "layer1.weight": torch.randn(128, 28),
            "layer1.bias": torch.randn(128),
        }

    @pytest.fixture
    def normal_update(self, global_params):
        """Create normal update."""
        return ModelUpdate(
            client_id="client_0",
            round_num=1,
            parameters={
                name: param + torch.randn_like(param) * 0.01
                for name, param in global_params.items()
            },
            num_samples=100,
            metrics={"loss": 0.5},
        )

    @pytest.fixture
    def anomalous_update(self, global_params):
        """Create anomalous update."""
        return ModelUpdate(
            client_id="client_malicious",
            round_num=1,
            parameters={
                name: param + torch.randn_like(param) * 10.0
                for name, param in global_params.items()
            },
            num_samples=100,
            metrics={"loss": 5.0},
        )

    def test_compute_score(self, detector, normal_update, global_params):
        """Test computing combined score."""
        score = detector.compute_score(normal_update, global_params)
        assert 0.0 <= score <= 1.0

    def test_compute_anomaly_score(self, detector, normal_update, global_params):
        """Test computing detailed anomaly score."""
        anomaly_score = detector.compute_anomaly_score(normal_update, global_params)
        
        assert hasattr(anomaly_score, "magnitude_score")
        assert hasattr(anomaly_score, "direction_score")
        assert hasattr(anomaly_score, "loss_score")
        assert hasattr(anomaly_score, "combined_score")
        
        # All scores should be in [0, 1]
        assert 0.0 <= anomaly_score.magnitude_score <= 1.0
        assert 0.0 <= anomaly_score.direction_score <= 1.0
        assert 0.0 <= anomaly_score.loss_score <= 1.0
        assert 0.0 <= anomaly_score.combined_score <= 1.0

    def test_is_anomalous(self, detector, normal_update, global_params):
        """Test anomaly detection."""
        anomaly_score = detector.compute_anomaly_score(normal_update, global_params)
        
        # Modify combined score to test threshold
        anomaly_score.combined_score = 0.8
        assert detector.is_anomalous(anomaly_score) is True
        
        anomaly_score.combined_score = 0.5
        assert detector.is_anomalous(anomaly_score) is False

    def test_update_statistics(self, detector, normal_update, global_params):
        """Test updating all detector statistics."""
        updates = [normal_update] * 5
        detector.update_statistics(updates, global_params)
        
        # All sub-detectors should have updated statistics
        mag_stats = detector.magnitude_detector.get_statistics()
        assert mag_stats["count"] == 5

    def test_get_detector_statistics(self, detector, normal_update, global_params):
        """Test getting statistics from all detectors."""
        updates = [normal_update] * 5
        detector.update_statistics(updates, global_params)
        
        stats = detector.get_detector_statistics()
        assert "magnitude" in stats
        assert "direction" in stats
        assert "loss" in stats

    def test_ensemble_methods(self, global_params):
        """Test different ensemble methods."""
        update = ModelUpdate(
            client_id="client_0",
            round_num=1,
            parameters={
                name: param + torch.randn_like(param) * 0.01
                for name, param in global_params.items()
            },
            num_samples=100,
            metrics={"loss": 0.5},
        )

        # Test weighted method
        detector_weighted = EnsembleDetector(ensemble_method="weighted")
        score_weighted = detector_weighted.compute_score(update, global_params)
        assert 0.0 <= score_weighted <= 1.0

        # Test voting method
        detector_voting = EnsembleDetector(ensemble_method="voting")
        score_voting = detector_voting.compute_score(update, global_params)
        assert 0.0 <= score_voting <= 1.0

        # Test max method
        detector_max = EnsembleDetector(ensemble_method="max")
        score_max = detector_max.compute_score(update, global_params)
        assert 0.0 <= score_max <= 1.0

        # Test min method
        detector_min = EnsembleDetector(ensemble_method="min")
        score_min = detector_min.compute_score(update, global_params)
        assert 0.0 <= score_min <= 1.0

    def test_invalid_ensemble_method(self, global_params):
        """Test invalid ensemble method raises error."""
        detector = EnsembleDetector(ensemble_method="invalid")
        update = ModelUpdate(
            client_id="client_0",
            round_num=1,
            parameters={
                name: param + torch.randn_like(param) * 0.01
                for name, param in global_params.items()
            },
            num_samples=100,
            metrics={"loss": 0.5},
        )
        with pytest.raises(ValueError):
            detector.compute_score(update, global_params)


class TestDetectionIntegration:
    """Integration tests for detection."""

    def test_multi_round_detection(self):
        """Test detection across multiple FL rounds."""
        detector = EnsembleDetector()
        global_params = {
            "layer1.weight": torch.randn(64, 28),
            "layer1.bias": torch.randn(64),
        }

        # Simulate 3 rounds
        for round_num in range(3):
            # Create normal client updates
            normal_updates = []
            for i in range(8):
                update = ModelUpdate(
                    client_id=f"honest_client_{i}",
                    round_num=round_num,
                    parameters={
                        name: param + torch.randn_like(param) * 0.01
                        for name, param in global_params.items()
                    },
                    num_samples=100,
                    metrics={"loss": 0.5 + np.random.randn() * 0.1},
                )
                normal_updates.append(update)

            # Create malicious update in round 2
            if round_num == 1:
                malicious_update = ModelUpdate(
                    client_id="malicious_client",
                    round_num=round_num,
                    parameters={
                        name: param + torch.randn_like(param) * 10.0
                        for name, param in global_params.items()
                    },
                    num_samples=100,
                    metrics={"loss": 5.0},
                )
                all_updates = normal_updates + [malicious_update]
            else:
                all_updates = normal_updates

            # Update statistics
            detector.update_statistics(all_updates, global_params)

            # Check detection for malicious update
            if round_num == 1:
                anomaly_score = detector.compute_anomaly_score(
                    all_updates[-1], global_params
                )
                # Malicious update should have higher anomaly score
                assert anomaly_score.combined_score > 0.1
