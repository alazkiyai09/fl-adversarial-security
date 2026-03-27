"""
Integration tests for FL anomaly detection system.
Tests ensemble voting and FL integration.
"""

import pytest
import numpy as np
from src.detectors import (
    MagnitudeDetector,
    SimilarityDetector,
    LayerwiseDetector,
    HistoricalDetector,
    ClusteringDetector,
    SpectralDetector
)
from src.ensemble.voting_ensemble import VotingEnsemble


class TestVotingEnsemble:
    """Tests for VotingEnsemble"""

    @pytest.fixture
    def honest_updates(self):
        """Generate synthetic honest client updates."""
        np.random.seed(42)
        updates = []
        for _ in range(10):
            update = np.random.randn(100) * 0.1
            updates.append(update)
        return updates

    @pytest.fixture
    def malicious_update(self):
        """Generate synthetic malicious update (large norm)."""
        np.random.seed(43)
        return np.random.randn(100) * 5.0

    @pytest.fixture
    def fitted_ensemble(self, honest_updates):
        """Create fitted ensemble with multiple detectors."""
        detectors = [
            MagnitudeDetector(method="zscore", threshold=2.0),
            SimilarityDetector(similarity_threshold=0.7),
            ClusteringDetector(method="isolation_forest", contamination=0.1)
        ]

        # Fit all detectors
        for detector in detectors:
            detector.fit(honest_updates)

        ensemble = VotingEnsemble(
            detectors=detectors,
            voting="majority"
        )

        return ensemble

    def test_ensemble_creation(self, honest_updates):
        """Test creating ensemble with multiple detectors."""
        detectors = [
            MagnitudeDetector(),
            SimilarityDetector()
        ]

        for detector in detectors:
            detector.fit(honest_updates)

        ensemble = VotingEnsemble(detectors=detectors)
        assert len(ensemble.detectors) == 2

    def test_majority_voting(self, fitted_ensemble, malicious_update):
        """Test majority voting strategy."""
        # Should be flagged by majority
        is_malicious = fitted_ensemble.is_malicious(malicious_update)
        assert is_malicious is True

    def test_soft_voting_score(self, fitted_ensemble, malicious_update, honest_updates):
        """Test soft voting (average of scores)."""
        fitted_ensemble.voting = "soft"

        malicious_score = fitted_ensemble.compute_anomaly_score(malicious_update)
        honest_score = fitted_ensemble.compute_anomaly_score(honest_updates[0])

        assert malicious_score > honest_score

    def test_get_individual_scores(self, fitted_ensemble, malicious_update):
        """Test getting individual detector scores."""
        scores = fitted_ensemble.get_individual_scores(malicious_update)

        assert len(scores) == len(fitted_ensemble.detectors)
        assert 'MagnitudeDetector' in scores
        assert 'SimilarityDetector' in scores
        assert 'ClusteringDetector' in scores

    def test_voting_summary(self, fitted_ensemble, malicious_update):
        """Test getting detailed voting summary."""
        summary = fitted_ensemble.get_voting_summary(malicious_update)

        assert 'votes' in summary
        assert 'scores' in summary
        assert 'num_flags' in summary
        assert 'ensemble_decision' in summary

    def test_empty_detectors_raises_error(self):
        """Test that empty detector list raises error."""
        with pytest.raises(ValueError):
            VotingEnsemble(detectors=[])

    def test_unanimous_voting(self, honest_updates):
        """Test unanimous voting strategy."""
        detectors = [
            MagnitudeDetector(threshold=3.0),
            SimilarityDetector(similarity_threshold=0.5)
        ]

        for detector in detectors:
            detector.fit(honest_updates)

        ensemble = VotingEnsemble(detectors=detectors, voting="unanimous")

        # Honest update should NOT be flagged unanimously
        is_malicious = ensemble.is_malicious(honest_updates[0])
        assert is_malicious is False


class TestDetectorIntegration:
    """Tests integrating multiple detector types"""

    @pytest.fixture
    def baseline_layered_updates(self):
        """Generate layer-wise baseline updates."""
        np.random.seed(42)
        updates = []
        for _ in range(10):
            update = {
                'layer_1': np.random.randn(50) * 0.1,
                'layer_2': np.random.randn(50) * 0.1,
                'layer_3': np.random.randn(50) * 0.1
            }
            updates.append(update)
        return updates

    def test_layerwise_detector_integration(self, baseline_layered_updates):
        """Test layer-wise detector with realistic updates."""
        detector = LayerwiseDetector(layer_threshold=2.0, min_anomalous_layers=2)
        detector.fit(baseline_layered_updates)

        # Create malicious update (one layer is very anomalous)
        malicious_update = {
            'layer_1': np.random.randn(50) * 5.0,  # Anomalous
            'layer_2': np.random.randn(50) * 0.1,   # Normal
            'layer_3': np.random.randn(50) * 5.0   # Anomalous
        }

        anomalous_layers = detector.get_anomalous_layers(malicious_update)
        assert len(anomalous_layers) >= 2

        is_malicious = detector.is_malicious(malicious_update)
        assert is_malicious is True

    def test_historical_detector_reputation_tracking(self):
        """Test historical detector reputation tracking."""
        detector = HistoricalDetector(alpha=0.3, threshold=2.0)
        detector.fit()

        client_id = "client_1"

        # Simulate multiple rounds
        for round_num in range(10):
            update = np.random.randn(100) * (0.5 if round_num < 7 else 5.0)
            score = abs(np.linalg.norm(update) - 1.0)
            detector.update_reputation(client_id, score)

        # Reputation should increase after malicious updates
        reputation = detector.get_reputation(client_id)
        assert reputation > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
