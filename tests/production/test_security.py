"""Unit tests for security module."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import numpy as np
import torch

from src.security.attack_detection import (
    PoisoningDetector,
    BackdoorDetector,
    LabelFlippingDetector,
    AttackDetector,
    AttackType,
)
from src.security.anomaly_logger import (
    AnomalyLogger,
    FLAnomalyLogger,
    AnomalySeverity,
    AnomalyType,
)
from src.security.alerting import (
    AlertManager,
    AlertChannel,
    AlertConfig,
    create_alert_manager,
)


@pytest.fixture
def sample_updates():
    """Create sample model updates."""
    updates = []

    # 4 benign updates (similar)
    for _ in range(4):
        update = [
            torch.randn(10, 10) * 0.1,  # Small random updates
            torch.randn(5) * 0.1,
        ]
        updates.append(update)

    # 1 malicious update (very different)
    updates.append([
        torch.randn(10, 10) * 10,  # Large magnitude
        torch.randn(5) * 10,
    ])

    return updates


class TestPoisoningDetector:
    """Tests for PoisoningDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = PoisoningDetector(threshold=3.0, method="statistical")

        assert detector.threshold == 3.0
        assert detector.method == "statistical"
        assert len(detector.update_magnitudes) == 0

    def test_statistical_detection(self, sample_updates):
        """Test statistical detection method."""
        detector = PoisoningDetector(threshold=2.0, method="statistical")
        results = detector.detect(sample_updates)

        assert len(results) == len(sample_updates)
        # Should detect the malicious update
        assert any(r.is_malicious for r in results)
        assert results[-1].is_malicious  # Last one is malicious

    def test_clustering_detection(self, sample_updates):
        """Test clustering detection method."""
        detector = PoisoningDetector(threshold=2.0, method="clustering")
        results = detector.detect(sample_updates)

        assert len(results) == len(sample_updates)
        # Should detect the outlier
        assert any(r.is_malicious for r in results)

    def test_isolation_forest_detection(self, sample_updates):
        """Test isolation forest detection method."""
        detector = PoisoningDetector(threshold=2.0, method="isolation_forest")
        results = detector.detect(sample_updates)

        assert len(results) == len(sample_updates)

    def test_hybrid_detection(self, sample_updates):
        """Test hybrid detection method."""
        detector = PoisoningDetector(threshold=2.0, method="hybrid")
        results = detector.detect(sample_updates)

        assert len(results) == len(sample_updates)
        # Should detect the malicious one
        malicious_count = sum(1 for r in results if r.is_malicious)
        assert malicious_count >= 1

    def test_all_benign_updates(self):
        """Test with all benign updates."""
        detector = PoisoningDetector(threshold=3.0)

        # All similar updates
        updates = [
            [torch.randn(10, 10) * 0.1, torch.randn(5) * 0.1]
            for _ in range(5)
        ]

        results = detector.detect(updates)

        # Should not detect any as malicious
        assert not any(r.is_malicious for r in results)


class TestBackdoorDetector:
    """Tests for BackdoorDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = BackdoorDetector(threshold=0.9)

        assert detector.threshold == 0.9
        assert detector.backdoor_test_set is None

    def test_detect_without_test_set(self, sample_updates):
        """Test detection without backdoor test set."""
        detector = BackdoorDetector(threshold=0.9)

        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )

        result = detector.detect(model, torch.device("cpu"))

        # Should return benign result without test set
        assert result.is_malicious is False
        assert "No backdoor test set provided" in result.details.get("message", "")


class TestLabelFlippingDetector:
    """Tests for LabelFlippingDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = LabelFlippingDetector(expected_fraud_rate=0.05, tolerance=0.1)

        assert detector.expected_fraud_rate == 0.05
        assert detector.tolerance == 0.1

    def test_normal_predictions(self):
        """Test with normal predictions."""
        detector = LabelFlippingDetector(expected_fraud_rate=0.05, tolerance=0.1)

        # Normal predictions
        predictions = np.random.binomial(1, 0.05, 100)
        labels = np.random.binomial(1, 0.05, 100)

        result = detector.detect(predictions, labels)

        assert result.is_malicious is False

    def test_flipped_predictions(self):
        """Test with flipped labels."""
        detector = LabelFlippingDetector(expected_fraud_rate=0.05, tolerance=0.1)

        # Flipped predictions (high fraud rate)
        predictions = np.ones(100, dtype=int)  # All fraud
        labels = np.random.binomial(1, 0.05, 100)

        result = detector.detect(predictions, labels)

        assert result.is_malicious is True
        assert result.attack_type == AttackType.LABEL_FLIPPING


class TestAttackDetector:
    """Tests for AttackDetector orchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        config = {
            "anomaly_threshold": 3.0,
            "method": "statistical",
            "expected_fraud_rate": 0.05,
            "tolerance": 0.1,
        }

        detector = AttackDetector(config)

        assert detector.poisoning_detector is not None
        assert detector.label_flipping_detector is not None

    def test_detect_poisoning(self, sample_updates):
        """Test poisoning detection."""
        config = {"anomaly_threshold": 2.0, "method": "statistical"}
        detector = AttackDetector(config)

        results = detector.detect_poisoning(sample_updates)

        assert len(results) == len(sample_updates)
        assert any(r.is_malicious for r in results)

    def test_detect_all_attacks(self, sample_updates):
        """Test comprehensive attack detection."""
        config = {"anomaly_threshold": 2.0, "method": "statistical"}
        detector = AttackDetector(config)

        results = detector.detect_all_attacks(sample_updates)

        assert "poisoning" in results
        assert "summary" in results
        assert results["summary"]["total_clients"] == len(sample_updates)


class TestAnomalyLogger:
    """Tests for AnomalyLogger."""

    def test_initialization(self):
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = AnomalyLogger(log_file=log_file)

            assert logger.log_file == log_file
            assert len(logger.events) == 0

    def test_log_event(self):
        """Test logging an event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            anomaly_logger = AnomalyLogger(log_file=log_file, console_output=False)

            event = anomaly_logger.log(
                anomaly_type=AnomalyType.POISONING_ATTACK,
                severity=AnomalySeverity.HIGH,
                message="Test anomaly",
                client_id=1,
                round_num=10,
                confidence=0.8,
            )

            assert event in anomaly_logger.events
            assert event.client_id == 1
            assert event.round_num == 10
            assert event.confidence == 0.8

            # Check file was created
            assert log_file.exists()

    def test_get_events_filtered(self):
        """Test filtering events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            anomaly_logger = AnomalyLogger(log_file=log_file, console_output=False)

            # Log multiple events
            anomaly_logger.log(
                AnomalyType.POISONING_ATTACK,
                AnomalySeverity.HIGH,
                "Test 1",
                client_id=1,
                round_num=10,
            )
            anomaly_logger.log(
                AnomalyType.BACKDOOR_ATTACK,
                AnomalySeverity.CRITICAL,
                "Test 2",
                client_id=2,
                round_num=10,
            )
            anomaly_logger.log(
                AnomalyType.POISONING_ATTACK,
                AnomalySeverity.MEDIUM,
                "Test 3",
                client_id=1,
                round_num=11,
            )

            # Filter by type
            poisoning_events = anomaly_logger.get_events(anomaly_type=AnomalyType.POISONING_ATTACK)
            assert len(poisoning_events) == 2

            # Filter by client
            client1_events = anomaly_logger.get_events(client_id=1)
            assert len(client1_events) == 2

            # Filter by round
            round10_events = anomaly_logger.get_events(round_num=10)
            assert len(round10_events) == 2

    def test_get_statistics(self):
        """Test statistics computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            anomaly_logger = AnomalyLogger(log_file=log_file, console_output=False)

            # Log events
            anomaly_logger.log(
                AnomalyType.POISONING_ATTACK,
                AnomalySeverity.HIGH,
                "Test",
                client_id=1,
            )
            anomaly_logger.log(
                AnomalyType.POISONING_ATTACK,
                AnomalySeverity.HIGH,
                "Test",
                client_id=2,
            )

            stats = anomaly_logger.get_statistics()

            assert "poisoning_attack_high" in stats
            assert stats["poisoning_attack_high"] == 2

    def test_get_summary(self):
        """Test summary computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            anomaly_logger = AnomalyLogger(log_file=log_file, console_output=False)

            # Log events
            for i in range(5):
                anomaly_logger.log(
                    AnomalyType.POISONING_ATTACK,
                    AnomalySeverity.HIGH,
                    f"Test {i}",
                )

            summary = anomaly_logger.get_summary()

            assert summary["total_events"] == 5
            assert "poisoning_attack" in summary["by_type"]
            assert summary["by_type"]["poisoning_attack"] == 5
            assert "high" in summary["by_severity"]

    def test_export_import(self):
        """Test exporting and importing events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            export_file = Path(tmpdir) / "export.json"

            # Create logger and log events
            logger1 = AnomalyLogger(log_file=log_file, console_output=False)
            logger1.log(
                AnomalyType.POISONING_ATTACK,
                AnomalySeverity.HIGH,
                "Test",
                client_id=1,
                round_num=10,
            )

            # Export
            logger1.export_to_file(export_file)
            assert export_file.exists()

            # Create new logger and import
            logger2 = AnomalyLogger(log_file=log_file, console_output=False)
            logger2.import_from_file(export_file)

            assert len(logger2.events) == 1
            assert logger2.events[0].client_id == 1


class TestFLAnomalyLogger:
    """Tests for FL-specific AnomalyLogger."""

    def test_log_poisoning_attack(self):
        """Test poisoning attack logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            anomaly_logger = FLAnomalyLogger(log_file=log_file, console_output=False)

            anomaly_logger.log_poisoning_attack(
                client_ids=[1, 2],
                round_num=10,
                confidence=0.8,
            )

            events = anomaly_logger.get_events(client_id=1)
            assert len(events) == 1
            assert events[0].anomaly_type == AnomalyType.POISONING_ATTACK

    def test_log_backdoor_attack(self):
        """Test backdoor attack logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            anomaly_logger = FLAnomalyLogger(log_file=log_file, console_output=False)

            anomaly_logger.log_backdoor_attack(
                round_num=10,
                confidence=0.9,
            )

            events = anomaly_logger.get_events(anomaly_type=AnomalyType.BACKDOOR_ATTACK)
            assert len(events) == 1
            assert events[0].severity == AnomalySeverity.CRITICAL

    def test_get_client_risk_score(self):
        """Test client risk score computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            anomaly_logger = FLAnomalyLogger(log_file=log_file, console_output=False)

            # Log some events for client 1
            anomaly_logger.log(
                AnomalyType.POISONING_ATTACK,
                AnomalySeverity.HIGH,
                "Test",
                client_id=1,
                confidence=0.8,
            )
            anomaly_logger.log(
                AnomalyType.UPDATE_OUTLIER,
                AnomalySeverity.MEDIUM,
                "Test",
                client_id=1,
                confidence=0.5,
            )

            risk_score = anomaly_logger.get_client_risk_score(1)

            assert 0.0 <= risk_score <= 1.0
            assert risk_score > 0.0  # Should have some risk

            # Client 2 should have no risk
            risk_score_2 = anomaly_logger.get_client_risk_score(2)
            assert risk_score_2 == 0.0


class TestAlertManager:
    """Tests for AlertManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = AlertManager()

        assert len(manager.channels) == 0

    def test_add_channel(self):
        """Test adding alert channel."""
        manager = AlertManager()

        config = AlertConfig(
            channel=AlertChannel.LOG,
            enabled=True,
            min_severity=AnomalySeverity.MEDIUM,
        )

        manager.add_channel(config)

        assert len(manager.channels) == 1

    def test_send_alert_log_channel(self):
        """Test sending alert to log channel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            anomaly_logger = AnomalyLogger(log_file=log_file, console_output=False)

            manager = AlertManager()
            config = AlertConfig(
                channel=AlertChannel.LOG,
                enabled=True,
                min_severity=AnomalySeverity.MEDIUM,
            )
            manager.add_channel(config)

            # Create event
            event = anomaly_logger.log(
                AnomalyType.POISONING_ATTACK,
                AnomalySeverity.HIGH,
                "Test alert",
                client_id=1,
                round_num=10,
            )

            # Send alert
            result = manager.send_alert(event)

            assert result is True

    def test_severity_filtering(self):
        """Test severity-based filtering."""
        manager = AlertManager()

        # Add channel with HIGH threshold
        config = AlertConfig(
            channel=AlertChannel.LOG,
            enabled=True,
            min_severity=AnomalySeverity.HIGH,
        )
        manager.add_channel(config)

        # Create LOW severity event
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            anomaly_logger = AnomalyLogger(log_file=log_file, console_output=False)

            event = anomaly_logger.log(
                AnomalyType.UPDATE_OUTLIER,
                AnomalySeverity.LOW,
                "Test",
            )

            # Should not send (below threshold)
            result = manager.send_alert(event)
            # Returns False due to severity filtering
            assert result is False

    def test_rate_limiting(self):
        """Test rate limiting."""
        manager = AlertManager(rate_limit_minutes=5)

        config = AlertConfig(
            channel=AlertChannel.LOG,
            enabled=True,
        )
        manager.add_channel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            anomaly_logger = AnomalyLogger(log_file=log_file, console_output=False)

            # Send first alert
            event1 = anomaly_logger.log(
                AnomalyType.POISONING_ATTACK,
                AnomalySeverity.HIGH,
                "Test 1",
            )
            result1 = manager.send_alert(event1)

            # Send second alert immediately (should be rate limited)
            event2 = anomaly_logger.log(
                AnomalyType.POISONING_ATTACK,
                AnomalySeverity.HIGH,
                "Test 2",
            )
            result2 = manager.send_alert(event2)

            # First should succeed, second should be rate limited
            assert result1 is True
            assert result2 is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
