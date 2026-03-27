"""
Unit Tests for SignGuard Anomaly Detection

Tests anomaly detection factors and main detector.
"""

import pytest
import numpy as np

from src.detection.factors import AnomalyFactors, AnomalyThreshold
from src.detection.anomaly_detector import AnomalyDetector
from src.detection.statistics import OnlineStatistics


class TestAnomalyFactors:
    """Test AnomalyFactors functionality."""

    def test_l2_anomaly_normal(self):
        """Test L2 anomaly with normal update."""
        factors = AnomalyFactors()

        # Create similar updates (normal case)
        median = np.random.randn(100, 100) * 0.1
        update = median + np.random.randn(100, 100) * 0.05  # Small deviation

        score = factors.compute_l2_anomaly(update, median)

        # Should be low (normal)
        assert 0.0 <= score <= 1.0
        assert score < 0.5

    def test_l2_anomaly_abnormal(self):
        """Test L2 anomaly with anomalous update."""
        factors = AnomalyFactors()

        median = np.random.randn(100, 100) * 0.1
        update = median + np.random.randn(100, 100) * 5.0  # Large deviation

        score = factors.compute_l2_anomaly(update, median)

        # Should be high (anomalous)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_directional_anomaly_aligned(self):
        """Test directional anomaly with aligned update."""
        factors = AnomalyFactors()

        # Same direction
        global_dir = np.random.randn(100)
        update = global_dir * 1.1  # Similar direction, slightly different magnitude

        score = factors.compute_directional_anomaly(update, global_dir)

        # Should be low (aligned)
        assert 0.0 <= score <= 1.0
        assert score < 0.5

    def test_directional_anomaly_opposite(self):
        """Test directional anomaly with opposite update."""
        factors = AnomalyFactors()

        # Opposite direction
        global_dir = np.random.randn(100)
        update = -global_dir * 1.0  # Exact opposite

        score = factors.compute_directional_anomaly(update, global_dir)

        # Should be high (opposite direction)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_layer_anomaly(self):
        """Test layer-wise anomaly detection."""
        factors = AnomalyFactors()

        # Create layer updates
        update = [
            np.random.randn(50, 50) * 0.1,  # Normal layer 1
            np.random.randn(50) * 2.0,       # Anomalous layer 2
            np.random.randn(50, 10) * 0.1    # Normal layer 3
        ]

        median = [
            np.random.randn(50, 50) * 0.1,
            np.random.randn(50) * 0.1,
            np.random.randn(50, 10) * 0.1
        ]

        layer_names = ["dense1", "dense2", "output"]

        scores = factors.compute_layer_anomaly(update, median, layer_names)

        # Should return scores for all layers
        assert len(scores) == 3
        assert "dense1" in scores
        assert "dense2" in scores
        assert "output" in scores

        # Layer 2 should be most anomalous
        assert scores["dense2"] > scores["dense1"]

    def test_temporal_anomaly_consistent(self):
        """Test temporal anomaly with consistent client."""
        factors = AnomalyFactors()

        # Consistent updates
        base_update = np.random.randn(100) * 0.1
        history = [base_update + np.random.randn(100) * 0.01 for _ in range(5)]
        current = base_update + np.random.randn(100) * 0.01

        score = factors.compute_temporal_anomaly(current, history)

        # Should be low (consistent)
        assert 0.0 <= score <= 1.0
        assert score < 0.5

    def test_temporal_anomaly_inconsistent(self):
        """Test temporal anomaly with sudden change."""
        factors = AnomalyFactors()

        # Sudden change in pattern
        history = [np.random.randn(100) * 0.1 for _ in range(5)]
        current = np.random.randn(100) * 5.0  # Large jump

        score = factors.compute_temporal_anomaly(current, history)

        # Should be high (inconsistent)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_combined_anomaly(self):
        """Test combined anomaly score computation."""
        factors = AnomalyFactors()

        median = np.random.randn(100) * 0.1
        global_dir = median.copy()
        update = median + np.random.randn(100) * 0.05  # Slightly anomalous

        scores = factors.compute_combined_anomaly(
            update=update,
            median_update=median,
            global_direction=global_dir,
            client_id="client_1",
            client_history=[],
            layer_names=None
        )

        # Check all factors present
        assert 'l2_magnitude' in scores
        assert 'directional_consistency' in scores
        assert 'layer_wise' in scores
        assert 'temporal_consistency' in scores
        assert 'combined' in scores

        # Combined should be weighted average
        assert 0.0 <= scores['combined'] <= 1.0


class TestAnomalyThreshold:
    """Test AnomalyThreshold functionality."""

    def test_percentile_threshold(self):
        """Test percentile-based threshold."""
        threshold = AnomalyThreshold(method='percentile', percentile=90)

        # Add scores
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold.update(scores)

        # 90th percentile should be 0.9
        assert abs(threshold.get_threshold() - 0.9) < 0.05

    def test_adaptive_threshold(self):
        """Test adaptive threshold."""
        threshold = AnomalyThreshold(method='adaptive')

        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold.update(scores)

        # Mean + 2*std
        mean = np.mean(scores)
        std = np.std(scores)
        expected = mean + 2 * std

        assert abs(threshold.get_threshold() - expected) < 0.01

    def test_is_anomalous(self):
        """Test anomaly detection with threshold."""
        threshold = AnomalyThreshold(method='percentile', percentile=80)

        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold.update(scores)

        # Score below threshold
        assert not threshold.is_anomalous(0.3)

        # Score above threshold
        assert threshold.is_anomalous(0.6)


class TestAnomalyDetector:
    """Test main AnomalyDetector."""

    def test_detect_anomalies(self):
        """Test anomaly detection across multiple clients."""
        detector = AnomalyDetector()

        # Create updates (10 clients, 1 anomalous)
        updates = {}
        for i in range(10):
            if i == 5:
                # Anomalous update
                updates[f"client_{i}"] = [
                    np.random.randn(100) * 5.0  # Large deviation
                ]
            else:
                # Normal update
                updates[f"client_{i}"] = [
                    np.random.randn(100) * 0.1
                ]

        # Detect anomalies
        results = detector.detect_anomalies(updates)

        # Should have results for all clients
        assert len(results) == 10

        # Client 5 should be most anomalous
        client_5_score = results["client_5"]["combined"]
        for client_id, result in results.items():
            if client_id != "client_5":
                assert client_5_score > result["combined"]

    def test_anomaly_summary(self):
        """Test anomaly summary computation."""
        detector = AnomalyDetector()

        # Create test results
        updates = {
            "client_0": [np.random.randn(100) * 0.1],
            "client_1": [np.random.randn(100) * 0.1],
            "client_2": [np.random.randn(100) * 5.0],  # Anomalous
        }

        results = detector.detect_anomalies(updates)
        summary = detector.get_anomaly_summary(results)

        # Check summary fields
        assert 'num_clients' in summary
        assert 'num_anomalous' in summary
        assert 'anomaly_rate' in summary
        assert 'mean_score' in summary
        assert 'std_score' in summary
        assert 'min_score' in summary
        assert 'max_score' in summary
        assert 'median_score' in summary

        assert summary['num_clients'] == 3
        assert summary['anomaly_rate'] > 0.0

    def test_top_anomalous_clients(self):
        """Test getting top anomalous clients."""
        detector = AnomalyDetector()

        updates = {
            "client_0": [np.random.randn(50) * 0.1],
            "client_1": [np.random.randn(50) * 2.0],  # Most anomalous
            "client_2": [np.random.randn(50) * 0.5],
            "client_3": [np.random.randn(50) * 1.0],
        }

        results = detector.detect_anomalies(updates)
        top_clients = detector.get_top_anomalous_clients(results, top_k=2)

        # Should return 2 clients
        assert len(top_clients) == 2

        # First should have highest score
        assert top_clients[0][1] > top_clients[1][1]

    def test_factor_weights(self):
        """Test setting factor weights."""
        detector = AnomalyDetector()

        new_weights = {
            'l2_magnitude': 0.5,
            'directional_consistency': 0.3,
            'layer_wise': 0.1,
            'temporal_consistency': 0.1
        }

        detector.set_factor_weights(new_weights)

        retrieved = detector.get_factor_weights()

        # Should be normalized
        assert abs(sum(retrieved.values()) - 1.0) < 1e-6

        # Ratios should be preserved
        assert abs(retrieved['l2_magnitude'] / retrieved['layer_wise'] - 5.0) < 0.01

    def test_detection_history(self):
        """Test detection history tracking."""
        detector = AnomalyDetector()

        updates = {
            "client_0": [np.random.randn(50) * 0.1],
        }

        # Run detection for 3 rounds
        for round_num in range(3):
            detector.detect_anomalies(updates)

        history = detector.get_detection_history()

        assert len(history) == 3
        assert history[0]['round'] == 0
        assert history[1]['round'] == 1
        assert history[2]['round'] == 2

    def test_reset(self):
        """Test resetting detector."""
        detector = AnomalyDetector()

        updates = {"client_0": [np.random.randn(50) * 0.1]}
        detector.detect_anomalies(updates)

        # Reset
        detector.reset()

        # History should be cleared
        assert len(detector.get_detection_history()) == 0


class TestOnlineStatistics:
    """Test OnlineStatistics integration."""

    def test_update_with_detector(self):
        """Test that detector updates online statistics."""
        detector = AnomalyDetector()

        updates = {
            "client_0": [np.random.randn(50) * 0.1],
            "client_1": [np.random.randn(50) * 0.1],
        }

        detector.detect_anomalies(updates)

        stats = detector.get_statistics()

        # Should have client history
        assert len(stats.get_client_history("client_0")) > 0
        assert len(stats.get_client_history("client_1")) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
