"""
Unit tests for Attack Detection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.servers.detection import AttackDetector


class TestAttackDetector:
    """Test attack detection mechanisms."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = AttackDetector(
            l2_norm_threshold=10.0,
            cosine_similarity_threshold=-0.5
        )

        assert detector.l2_norm_threshold == 10.0
        assert detector.cosine_threshold == -0.5

    def test_l2_norm_computation(self):
        """Test L2 norm computation."""
        detector = AttackDetector()

        parameters = np.array([3.0, 4.0])  # Should have L2 norm = 5
        l2 = detector.compute_l2_norm(parameters)

        np.testing.assert_almost_equal(l2, 5.0)

    def test_cosine_similarity_identical(self):
        """Test cosine similarity for identical vectors."""
        detector = AttackDetector()

        update = np.array([1.0, 2.0, 3.0])
        similarity = detector.compute_cosine_similarity(update, update)

        # Cosine similarity of identical vectors is 1.0
        np.testing.assert_almost_equal(similarity, 1.0)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity for opposite vectors."""
        detector = AttackDetector()

        update_a = np.array([1.0, 2.0, 3.0])
        update_b = -update_a

        similarity = detector.compute_cosine_similarity(update_a, update_b)

        # Cosine similarity of opposite vectors is -1.0
        np.testing.assert_almost_equal(similarity, -1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors."""
        detector = AttackDetector()

        update_a = np.array([1.0, 0.0])
        update_b = np.array([0.0, 1.0])

        similarity = detector.compute_cosine_similarity(update_a, update_b)

        # Cosine similarity of orthogonal vectors is 0.0
        np.testing.assert_almost_equal(similarity, 0.0)

    def test_detect_l2_outliers(self):
        """Test L2 norm outlier detection."""
        detector = AttackDetector(std_threshold=2.0)

        # Create normal updates and one outlier
        normal_updates = [np.random.randn(10) * 0.1 for _ in range(5)]
        outlier = np.random.randn(10) * 10.0  # Large magnitude

        all_updates = normal_updates + [outlier]
        client_ids = list(range(len(all_updates)))

        l2_norms = [detector.compute_l2_norm(u) for u in all_updates]
        outliers = detector._detect_l2_outliers(l2_norms, client_ids)

        # Last client (outlier) should be detected
        assert client_ids[-1] in outliers

    def test_detect_sign_flipping(self):
        """Test detection of sign flipping attack via cosine similarity."""
        detector = AttackDetector(cosine_similarity_threshold=-0.5)

        np.random.seed(42)
        # Create a base direction
        base_direction = np.random.randn(10)

        # Create honest updates that all point in the same direction (with small noise)
        honest_updates = [base_direction + np.random.randn(10) * 0.1 for _ in range(5)]

        # Create sign-flipped update
        malicious_update = -base_direction

        all_updates = honest_updates + [malicious_update]
        client_ids = list(range(len(all_updates)))

        cosine_matrix = detector._compute_cosine_matrix(all_updates)

        # Malicious update should have negative correlation with honest updates
        # Check similarity with first honest update
        sim_with_first = cosine_matrix[-1, 0]
        assert sim_with_first < -0.8  # Strong negative correlation

    def test_detect_anomalies_integration(self):
        """Test full anomaly detection pipeline."""
        np.random.seed(42)
        detector = AttackDetector(
            l2_norm_threshold=5.0,
            cosine_similarity_threshold=-0.5,
            std_threshold=2.0
        )

        # Mix of honest and malicious clients
        honest1 = np.random.randn(10) * 0.1
        honest2 = np.random.randn(10) * 0.1
        large_norm = np.random.randn(10) * 10.0  # Malicious (large L2 norm)
        sign_flip = -honest1  # Malicious (sign flip)

        updates = [honest1, honest2, large_norm, sign_flip]
        client_ids = list(range(len(updates)))

        result = detector.detect_anomalies(updates, client_ids)

        # Large norm attack should be detected via L2 norm
        l2_norms = [detector.compute_l2_norm(u) for u in updates]
        assert l2_norms[2] > 10.0  # Verify large norm

        # Check detection details are computed
        assert "detection_details" in result
        assert "l2_norms" in result["detection_details"]

    def test_empty_updates(self):
        """Test detection with no updates."""
        detector = AttackDetector()

        result = detector.detect_anomalies([], [])

        assert result["suspicious_clients"] == []
        assert result["detection_details"] == {}

    def test_detection_statistics(self):
        """Test detection statistics computation."""
        detector = AttackDetector()

        # Simulate multiple rounds
        for _ in range(3):
            updates = [np.random.randn(10) for _ in range(5)]
            detector.detect_anomalies(updates, list(range(5)))

        stats = detector.get_detection_statistics()

        assert stats["total_detections"] == 3
        assert "avg_suspicious_per_round" in stats
