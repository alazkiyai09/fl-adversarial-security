"""
Data fetching utilities with caching for efficient updates.
Handles data retrieval from Redis, WebSocket, or demo data.
"""

import streamlit as st
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from src.dashboard.app.utils.session import get_metrics_collector


# Demo data directory
DEMO_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "demo_scenarios"


@st.cache_data(ttl=1)  # Cache for 1 second
def get_training_summary() -> Dict[str, Any]:
    """
    Get summary of training progress.

    Returns:
        Dictionary with training summary
    """
    mc = get_metrics_collector()
    return mc.get_summary_statistics()


@st.cache_data(ttl=1)
def get_recent_rounds(n: int = 10) -> List:
    """
    Get recent training rounds.

    Args:
        n: Number of recent rounds

    Returns:
        List of recent TrainingRound objects
    """
    mc = get_metrics_collector()
    return mc.get_recent_rounds(n)


@st.cache_data(ttl=1)
def get_client_metrics_summary() -> Dict[int, Dict]:
    """
    Get summary of client metrics.

    Returns:
        Dictionary mapping client_id to metrics summary
    """
    mc = get_metrics_collector()
    client_data = mc.get_client_metrics(n_recent=1)

    summary = {}
    for client_id, metrics in client_data.items():
        if metrics:
            latest = metrics[-1]
            summary[client_id] = {
                "accuracy": latest.accuracy,
                "loss": latest.loss,
                "status": latest.status,
                "reputation": latest.reputation_score,
                "anomaly_score": latest.anomaly_score
            }

    return summary


@st.cache_data(ttl=1)
def get_security_events(n: int = 20) -> List:
    """
    Get recent security events.

    Args:
        n: Number of recent events

    Returns:
        List of SecurityEvent objects
    """
    mc = get_metrics_collector()
    return mc.get_security_events(n_recent=n)


@st.cache_data(ttl=1)
def get_client_reputations() -> Dict[int, float]:
    """Get current client reputation scores."""
    mc = get_metrics_collector()
    return mc.get_client_reputations()


@st.cache_data(ttl=1)
def get_attack_statistics() -> Dict[str, Any]:
    """Get attack detection statistics."""
    mc = get_metrics_collector()
    return mc.get_attack_statistics()


@st.cache_data(ttl=1)
def get_privacy_status() -> Optional[Dict]:
    """Get privacy budget status."""
    mc = get_metrics_collector()
    return mc.get_privacy_status()


def load_demo_scenario(scenario_name: str) -> None:
    """
    Load a demo scenario from file.

    Args:
        scenario_name: Name of scenario to load
    """
    # Map scenario names to file names
    scenario_files = {
        "Normal Training": "normal_training.json",
        "Label Flipping Attack": "label_flipping.json",
        "Backdoor Attack": "backdoor.json",
        "Byzantine Attack": "byzantine.json",
        "SignGuard Defense": "signguard_defense.json",
        "FoolsGold Defense": "foolsgold_defense.json"
    }

    filename = scenario_files.get(scenario_name)
    if not filename:
        st.error(f"Unknown scenario: {scenario_name}")
        return

    filepath = DEMO_DATA_DIR / filename

    if not filepath.exists():
        st.warning(f"Demo scenario file not found: {filename}")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Load data into metrics collector
    mc = get_metrics_collector()
    mc.reset()

    # Convert JSON data back to objects
    from core.data_models import TrainingRound, ClientMetric, SecurityEvent
    from datetime import datetime

    for round_data in data.get("rounds", []):
        # Reconstruct client metrics
        client_metrics = [
            ClientMetric(**cm) for cm in round_data.get("per_client_metrics", [])
        ]

        # Reconstruct security events
        security_events = [
            SecurityEvent(**se) for se in round_data.get("security_events", [])
        ]

        # Create training round
        round_dict = {
            **round_data,
            "per_client_metrics": client_metrics,
            "security_events": security_events
        }
        training_round = TrainingRound(**round_dict)

        mc.add_round(training_round)

    st.session_state.current_round = len(data.get("rounds", []))
    st.success(f"Loaded {len(data.get('rounds', []))} rounds from {scenario_name}")


def generate_demo_data(num_rounds: int = 50) -> None:
    """
    Generate synthetic demo data for testing.

    Args:
        num_rounds: Number of rounds to generate
    """
    import numpy as np
    from core.data_models import TrainingRound, ClientMetric, SecurityEvent
    from datetime import datetime

    mc = get_metrics_collector()
    mc.reset()

    num_clients = 10

    for round_num in range(1, num_rounds + 1):
        # Generate improving metrics
        base_accuracy = 0.5 + 0.4 * (1 - np.exp(-0.05 * round_num))
        base_loss = 2.0 * np.exp(-0.03 * round_num)

        # Add some noise
        accuracy = base_accuracy + np.random.randn() * 0.02
        loss = base_loss + np.random.randn() * 0.05

        # Generate client metrics
        client_metrics = []
        for client_id in range(num_clients):
            client_acc = accuracy + np.random.randn() * 0.05
            client_loss = loss + np.random.randn() * 0.1
            anomaly_score = np.random.rand() * 0.2  # Low anomaly normally

            # Occasional anomalies
            if np.random.rand() < 0.05:
                anomaly_score = 0.6 + np.random.rand() * 0.4

            client_metrics.append(ClientMetric(
                client_id=client_id,
                accuracy=np.clip(client_acc, 0, 1),
                loss=max(0, client_loss),
                data_size=np.random.randint(500, 1500),
                training_time=np.random.uniform(1, 5),
                status="anomaly" if anomaly_score > 0.5 else "active",
                anomaly_score=anomaly_score,
                update_norm=np.random.uniform(0.5, 2.0),
                reputation_score=1.0 - anomaly_score * 0.5
            ))

        # Generate occasional security events
        security_events = []
        if round_num > 10 and np.random.rand() < 0.1:
            event_types = ["attack_detected", "anomaly_detected", "defense_activated"]
            security_events.append(SecurityEvent(
                event_id=f"event_{round_num}_{np.random.randint(1000)}",
                event_type=np.random.choice(event_types),
                severity=np.random.choice(["low", "medium", "high"]),
                message=f"Security event in round {round_num}",
                round_num=round_num,
                affected_clients=[np.random.randint(0, num_clients)],
                confidence=0.7 + np.random.rand() * 0.3
            ))

        training_round = TrainingRound(
            round_num=round_num,
            global_accuracy=np.clip(accuracy, 0, 1),
            global_loss=max(0, loss),
            per_client_metrics=client_metrics,
            security_events=security_events,
            epsilon_spent=0.1 if round_num % 5 == 0 else 0.0
        )

        mc.add_round(training_round)

    st.session_state.current_round = num_rounds
    st.success(f"Generated {num_rounds} rounds of demo data")


def clear_cache() -> None:
    """Clear all cached data."""
    st.cache_data.clear()


def should_refresh() -> bool:
    """
    Check if it's time to refresh based on refresh rate.

    Returns:
        True if should refresh
    """
    last_update = st.session_state.get("last_update_time", 0)
    refresh_rate = st.session_state.get("refresh_rate", 2000) / 1000  # Convert to seconds

    current_time = time.time()
    if current_time - last_update >= refresh_rate:
        st.session_state.last_update_time = current_time
        return True

    return False


def wait_for_refresh() -> None:
    """Wait for next refresh cycle if auto-refresh is enabled."""
    if st.session_state.get("auto_refresh", True):
        refresh_rate = st.session_state.get("refresh_rate", 2000)
        time.sleep(refresh_rate / 1000)
        st.rerun()


def _load_demo_scenario(scenario_name: str) -> None:
    """Helper function to load demo scenario."""
    load_demo_scenario(scenario_name)
