"""
Tests for core data models.
"""

import pytest
from core.data_models import (
    ClientMetric,
    TrainingRound,
    SecurityEvent,
    FLConfig,
    AttackConfig,
    DefenseConfig,
    PrivacyBudget
)


def test_client_metric_creation():
    """Test creating a ClientMetric."""
    metric = ClientMetric(
        client_id=0,
        accuracy=0.95,
        loss=0.05,
        data_size=1000,
        training_time=2.5,
        status="active"
    )

    assert metric.client_id == 0
    assert metric.accuracy == 0.95
    assert metric.loss == 0.05
    assert metric.status == "active"
    assert metric.anomaly_score == 0.0
    assert metric.reputation_score == 1.0


def test_training_round_creation():
    """Test creating a TrainingRound."""
    client_metrics = [
        ClientMetric(client_id=i, accuracy=0.9, loss=0.1, data_size=100, training_time=1.0)
        for i in range(5)
    ]

    round_data = TrainingRound(
        round_num=1,
        global_accuracy=0.92,
        global_loss=0.08,
        per_client_metrics=client_metrics
    )

    assert round_data.round_num == 1
    assert len(round_data.per_client_metrics) == 5
    assert round_data.global_accuracy == 0.92


def test_security_event_creation():
    """Test creating a SecurityEvent."""
    event = SecurityEvent(
        event_id="test_event",
        event_type="attack_detected",
        severity="high",
        message="Test attack detected",
        round_num=5,
        affected_clients=[1, 2]
    )

    assert event.event_id == "test_event"
    assert event.event_type == "attack_detected"
    assert event.severity == "high"
    assert len(event.affected_clients) == 2


def test_fl_config():
    """Test FL configuration."""
    config = FLConfig(
        num_rounds=100,
        num_clients=10,
        learning_rate=0.01
    )

    assert config.num_rounds == 100
    assert config.num_clients == 10
    assert config.learning_rate == 0.01


def test_attack_config():
    """Test attack configuration."""
    config = AttackConfig(
        attack_type="label_flipping",
        start_round=10,
        num_attackers=2
    )

    assert config.attack_type == "label_flipping"
    assert config.start_round == 10
    assert config.num_attackers == 2


def test_defense_config():
    """Test defense configuration."""
    config = DefenseConfig(
        defense_type="signguard",
        anomaly_threshold=0.5
    )

    assert config.defense_type == "signguard"
    assert config.anomaly_threshold == 0.5


def test_privacy_budget():
    """Test privacy budget tracking."""
    budget = PrivacyBudget(
        epsilon_total=10.0,
        epsilon_spent=2.5
    )

    assert budget.epsilon_total == 10.0
    assert budget.epsilon_spent == 2.5
    assert budget.epsilon_remaining == 7.5


def test_client_metric_validation():
    """Test ClientMetric validation."""
    # Update norm should be capped at 1000
    metric = ClientMetric(
        client_id=0,
        accuracy=0.9,
        loss=0.1,
        data_size=100,
        training_time=1.0,
        update_norm=2000  # Should be capped to 1000
    )

    assert metric.update_norm == 1000


def test_training_round_duplicate_clients():
    """Test that duplicate client IDs are rejected."""
    client_metrics = [
        ClientMetric(client_id=0, accuracy=0.9, loss=0.1, data_size=100, training_time=1.0),
        ClientMetric(client_id=0, accuracy=0.8, loss=0.2, data_size=100, training_time=1.0)
    ]

    with pytest.raises(ValueError, match="Duplicate client IDs"):
        TrainingRound(
            round_num=1,
            global_accuracy=0.85,
            global_loss=0.15,
            per_client_metrics=client_metrics
        )
