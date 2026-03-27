"""
Session state management for Streamlit.
Handles persistent state across page reloads.
"""

import streamlit as st
from typing import Optional
from core.data_models import FLConfig, AttackConfig, DefenseConfig
from core.metrics_collector import MetricsCollector
from src.dashboard.app.utils.config import (
    create_default_config,
    create_default_attack_config,
    create_default_defense_config
)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""

    # Training control
    if "training_running" not in st.session_state:
        st.session_state.training_running = False

    if "training_paused" not in st.session_state:
        st.session_state.training_paused = False

    if "current_round" not in st.session_state:
        st.session_state.current_round = 0

    # Configuration
    if "fl_config" not in st.session_state:
        st.session_state.fl_config = create_default_config()

    if "attack_config" not in st.session_state:
        st.session_state.attack_config = create_default_attack_config()

    if "defense_config" not in st.session_state:
        st.session_state.defense_config = create_default_defense_config()

    # Metrics collector
    if "metrics_collector" not in st.session_state:
        st.session_state.metrics_collector = MetricsCollector()

    # Experiment tracking
    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    if "current_experiment_id" not in st.session_state:
        st.session_state.current_experiment_id = None

    # Attack state
    if "attack_active" not in st.session_state:
        st.session_state.attack_active = False

    if "attack_injected" not in st.session_state:
        st.session_state.attack_injected = False

    # UI settings
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = True

    if "refresh_rate" not in st.session_state:
        st.session_state.refresh_rate = 2000

    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True

    # Data cache
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = None


def start_training() -> None:
    """Start training process."""
    st.session_state.training_running = True
    st.session_state.training_paused = False
    st.session_state.current_round = 0
    st.session_state.attack_injected = False

    # Reset metrics collector
    st.session_state.metrics_collector.reset()

    # Create new experiment ID
    import uuid
    st.session_state.current_experiment_id = f"exp_{uuid.uuid4().hex[:8]}"


def stop_training() -> None:
    """Stop training process."""
    st.session_state.training_running = False
    st.session_state.training_paused = False


def pause_training() -> None:
    """Pause training process."""
    st.session_state.training_paused = True


def resume_training() -> None:
    """Resume training process."""
    st.session_state.training_paused = False


def is_training_active() -> bool:
    """Check if training is currently active (running and not paused)."""
    return (
        st.session_state.get("training_running", False) and
        not st.session_state.get("training_paused", False)
    )


def update_fl_config(updates: dict) -> None:
    """
    Update FL configuration.

    Args:
        updates: Dictionary of fields to update
    """
    current = st.session_state.fl_config.model_dump()
    current.update(updates)
    st.session_state.fl_config = FLConfig(**current)


def update_attack_config(updates: dict) -> None:
    """Update attack configuration."""
    current = st.session_state.attack_config.model_dump()
    current.update(updates)
    st.session_state.attack_config = AttackConfig(**current)


def update_defense_config(updates: dict) -> None:
    """Update defense configuration."""
    current = st.session_state.defense_config.model_dump()
    current.update(updates)
    st.session_state.defense_config = DefenseConfig(**current)


def get_metrics_collector() -> MetricsCollector:
    """Get the current metrics collector instance."""
    return st.session_state.metrics_collector


def increment_round() -> int:
    """Increment round number and return new value."""
    st.session_state.current_round += 1
    return st.session_state.current_round


def inject_attack() -> None:
    """Mark attack as injected (for demo mode)."""
    st.session_state.attack_active = True
    st.session_state.attack_injected = True


def deactivate_attack() -> None:
    """Deactivate current attack."""
    st.session_state.attack_active = False


def reset_experiment() -> None:
    """Reset experiment state."""
    st.session_state.training_running = False
    st.session_state.training_paused = False
    st.session_state.current_round = 0
    st.session_state.attack_active = False
    st.session_state.attack_injected = False
    st.session_state.metrics_collector.reset()
    st.session_state.current_experiment_id = None


def get_experiment_summary() -> dict:
    """Get summary of current experiment."""
    mc = st.session_state.metrics_collector

    return {
        "experiment_id": st.session_state.current_experiment_id,
        "round": st.session_state.current_round,
        "running": st.session_state.training_running,
        "paused": st.session_state.training_paused,
        "attack_active": st.session_state.attack_active,
        "config": {
            "fl": st.session_state.fl_config.model_dump(),
            "attack": st.session_state.attack_config.model_dump(),
            "defense": st.session_state.defense_config.model_dump()
        }
    }


def save_experiment_to_history(name: str) -> None:
    """
    Save completed experiment to history.

    Args:
        name: Name for the experiment
    """
    if st.session_state.current_experiment_id is None:
        return

    mc = st.session_state.metrics_collector
    if not mc.training_history:
        return

    result = mc.export_experiment_result(
        st.session_state.current_experiment_id,
        name
    )

    st.session_state.experiments.append(result.model_dump())
