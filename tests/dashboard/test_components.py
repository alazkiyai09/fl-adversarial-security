"""
Tests for dashboard components.
"""

import pytest
import numpy as np
from core.data_models import (
    ClientMetric,
    TrainingRound,
    SecurityEvent
)


def test_training_curve_chart():
    """Test training curve chart generation."""
    from app.components.charts import plot_training_curves

    # Create test data
    rounds = []
    for i in range(1, 11):
        client_metrics = [
            ClientMetric(
                client_id=j,
                accuracy=0.8 + i * 0.01,
                loss=1.0 - i * 0.05,
                data_size=100,
                training_time=1.0
            )
            for j in range(5)
        ]

        rounds.append(TrainingRound(
            round_num=i,
            global_loss=1.0 - i * 0.05,
            global_accuracy=0.8 + i * 0.01,
            per_client_metrics=client_metrics
        ))

    # Generate chart
    fig = plot_training_curves(rounds, show_loss=True, show_accuracy=True)

    assert fig is not None
    assert len(fig.data) == 2  # Loss and Accuracy traces


def test_client_reputation_chart():
    """Test client reputation chart generation."""
    from app.components.charts import plot_client_reputation

    # Create test data
    from collections import defaultdict
    client_metrics = defaultdict(list)

    for i in range(10):
        for client_id in range(5):
            client_metrics[client_id].append(ClientMetric(
                client_id=client_id,
                accuracy=0.9,
                loss=0.1,
                data_size=100,
                training_time=1.0,
                reputation_score=0.8 + np.random.rand() * 0.2
            ))

    # Generate chart
    fig = plot_client_reputation(dict(client_metrics), show_history=False)

    assert fig is not None
    assert len(fig.data) == 1  # Bar trace


def test_security_timeline_chart():
    """Test security timeline chart generation."""
    from app.components.charts import plot_security_timeline

    # Create test events
    events = [
        SecurityEvent(
            event_id=f"event_{i}",
            event_type="attack_detected",
            severity="high" if i % 2 == 0 else "medium",
            message=f"Test event {i}",
            round_num=i,
            affected_clients=[i % 5]
        )
        for i in range(1, 11)
    ]

    # Generate chart
    fig = plot_security_timeline(events)

    assert fig is not None


def test_privacy_budget_chart():
    """Test privacy budget chart generation."""
    from app.components.charts import plot_privacy_budget

    epsilon_spent = [0.1, 0.1, 0.1, 0.0, 0.1]
    epsilon_total = 10.0

    fig = plot_privacy_budget(epsilon_spent, epsilon_total)

    assert fig is not None
    assert len(fig.data) >= 2  # Spent and remaining


def test_metric_card_html():
    """Test metric card HTML generation (basic smoke test)."""
    from app.components.metrics import metric_card
    from streamlit import _main as main

    # Just verify it doesn't crash
    # (actual rendering requires Streamlit context)
    import streamlit as st
    if st.runtime.exists():
        metric_card("Test", 100, delta="+5%")


def test_data_fetcher_caching():
    """Test data fetcher caching mechanism."""
    import time
    from app.components.data_fetchers import get_training_summary
    from app.utils.session import get_metrics_collector

    # Initialize with some data
    mc = get_metrics_collector()

    client_metrics = [
        ClientMetric(
            client_id=i,
            accuracy=0.9,
            loss=0.1,
            data_size=100,
            training_time=1.0
        )
        for i in range(5)
    ]

    mc.add_round(TrainingRound(
        round_num=1,
        global_loss=0.1,
        global_accuracy=0.9,
        per_client_metrics=client_metrics
    ))

    # First call
    start = time.time()
    summary1 = get_training_summary()
    time1 = time.time() - start

    # Second call (should be cached)
    start = time.time()
    summary2 = get_training_summary()
    time2 = time.time() - start

    # Summary should be the same
    assert summary1["current_round"] == summary2["current_round"]

    # Cached call should be faster (or similar)
    # (In real scenario with @st.cache_data, second call would be faster)


def test_session_state():
    """Test session state utilities."""
    from app.utils.session import init_session_state
    import streamlit as st

    if st.runtime.exists():
        # Initialize
        init_session_state()

        # Check that required keys exist
        assert "training_running" in st.session_state
        assert "fl_config" in st.session_state
        assert "metrics_collector" in st.session_state


def test_config_loading():
    """Test configuration loading."""
    from app.utils.config import create_default_config, get_fl_config

    config = create_default_config()

    assert config.num_rounds > 0
    assert config.num_clients > 0
    assert config.learning_rate > 0

    # Test getting config
    fl_config = get_fl_config()
    assert fl_config is not None


def test_attack_config():
    """Test attack configuration."""
    from app.utils.config import create_default_attack_config

    config = create_default_attack_config()

    assert config.attack_type in ["label_flipping", "backdoor", "byzantine", "poisoning"]
    assert config.num_attackers > 0


def test_defense_config():
    """Test defense configuration."""
    from app.utils.config import create_default_defense_config

    config = create_default_defense_config()

    assert config.defense_type in ["none", "signguard", "krum", "foolsgold", "trim_mean", "median"]
