"""
Unit tests for FL anomaly detectors.
Tests each detection method with synthetic honest and malicious updates.
"""

import pytest
import numpy as np
from src.detectors.magnitude_detector import MagnitudeDetector
from src.detectors.similarity_detector import SimilarityDetector


class TestMagnitudeDetector:
    """Tests for MagnitudeDetector"""

    @pytest.fixture
    def honest_updates(self):
        """Generate synthetic honest client updates."""
        np.random.seed(42)
        # 10 honest clients with similar norm (~1.0)
        updates = []
        for _ in range(10):
            update = np.random.randn(100) * 0.1
            updates.append(update / np.linalg.norm(update))  # Normalize to unit norm
        return updates

    @pytest.fixture
    def malicious_updates(self):
        """Generate synthetic malicious client updates."""
        np.random.seed(43)
        updates = []

        # Type 1: Large magnitude (5x normal)
        large_update = np.random.randn(100) * 0.5
        updates.append(large_update)

        # Type 2: Tiny magnitude (0.1x normal)
        small_update = np.random.randn(100) * 0.01
        updates.append(small_update)

        return updates

    def test_fit_zscore(self, honest_updates):
        """Test fitting with z-score method."""
        detector = MagnitudeDetector(method="zscore", threshold=3.0)
        detector.fit(honest_updates)

        assert detector.is_fitted
        assert detector.mean_norm > 0
        assert detector.std_norm > 0

    def test_fit_iqr(self, honest_updates):
        """Test fitting with IQR method."""
        detector = MagnitudeDetector(method="iqr", iqr_multiplier=1.5)
        detector.fit(honest_updates)

        assert detector.is_fitted
        assert detector.q1_norm > 0
        assert detector.q3_norm > 0

    def test_detect_large_magnitude(self, honest_updates, malicious_updates):
        """Test detecting updates with large magnitude."""
        detector = MagnitudeDetector(method="zscore", threshold=2.0)
        detector.fit(honest_updates)

        # Large magnitude should be flagged
        large_score = detector.compute_anomaly_score(malicious_updates[0])
        assert large_score > detector.threshold

    def test_honest_not_flagged(self, honest_updates):
        """Test that honest updates are not flagged."""
        detector = MagnitudeDetector(method="zscore", threshold=3.0)
        detector.fit(honest_updates)

        for update in honest_updates[:3]:  # Test a few
            score = detector.compute_anomaly_score(update)
            assert score < detector.threshold

    def test_empty_baseline_raises_error(self):
        """Test that empty baseline raises error."""
        detector = MagnitudeDetector()
        with pytest.raises(ValueError):
            detector.fit([])

    def test_score_before_fit_raises_error(self):
        """Test that scoring before fitting raises error."""
        detector = MagnitudeDetector()
        with pytest.raises(RuntimeError):
            detector.compute_anomaly_score(np.random.randn(100))


class TestSimilarityDetector:
    """Tests for SimilarityDetector"""

    @pytest.fixture
    def honest_updates(self):
        """Generate synthetic honest client updates."""
        np.random.seed(42)
        # 10 honest clients with similar direction
        base_direction = np.random.randn(100)
        base_direction = base_direction / np.linalg.norm(base_direction)

        updates = []
        for _ in range(10):
            noise = np.random.randn(100) * 0.1
            update = base_direction + noise
            updates.append(update)
        return updates

    @pytest.fixture
    def malicious_updates(self):
        """Generate synthetic malicious client updates."""
        np.random.seed(43)
        updates = []

        # Type 1: Opposite direction
        opposite = -np.random.randn(100)
        updates.append(opposite)

        # Type 2: Orthogonal direction
        orthogonal = np.random.randn(100)
        updates.append(orthogonal)

        return updates

    def test_fit(self, honest_updates):
        """Test fitting similarity detector."""
        detector = SimilarityDetector(similarity_threshold=0.8)
        detector.fit(honest_updates)

        assert detector.is_fitted
        assert detector.reference_update is not None

    def test_detect_opposite_direction(self, honest_updates, malicious_updates):
        """Test detecting updates with opposite direction."""
        detector = SimilarityDetector(similarity_threshold=0.5)
        detector.fit(honest_updates)

        # Opposite direction should have low similarity
        score = detector.compute_anomaly_score(malicious_updates[0])
        similarity = detector.get_similarity(malicious_updates[0])

        assert similarity < 0  # Negative similarity (opposite)
        assert score > 0  # High anomaly score

    def test_honest_high_similarity(self, honest_updates):
        """Test that honest updates have high similarity."""
        detector = SimilarityDetector(similarity_threshold=0.7)
        detector.fit(honest_updates)

        for update in honest_updates[:3]:
            similarity = detector.get_similarity(update)
            score = detector.compute_anomaly_score(update)

            assert similarity > 0.7  # High similarity
            assert score == 0.0  # Low anomaly score (not anomalous)

    def test_cosine_similarity_bounds(self):
        """Test that cosine similarity is in [-1, 1]."""
        detector = SimilarityDetector()

        a = np.random.randn(50)
        b = np.random.randn(50)

        similarity = detector._cosine_similarity(a, b)
        assert -1 <= similarity <= 1

    def test_zero_vector_handling(self):
        """Test handling of zero vectors."""
        detector = SimilarityDetector()

        zero_vec = np.zeros(10)
        normal_vec = np.random.randn(10)

        # Should return 0 for zero vector
        similarity = detector._cosine_similarity(zero_vec, normal_vec)
        assert similarity == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
