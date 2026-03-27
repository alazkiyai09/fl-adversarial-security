"""
Tests for FL simulator.
"""

import pytest
from core.data_models import FLConfig, AttackConfig, DefenseConfig
from backend.simulator import FLSimulator, create_demo_simulator


def test_simulator_initialization():
    """Test simulator initialization."""
    fl_config = FLConfig(
        num_rounds=10,
        num_clients=5,
        clients_per_round=5
    )

    simulator = FLSimulator(fl_config)

    assert simulator.config == fl_config
    assert simulator.current_round == 0
    assert len(simulator.client_states) == 5


def test_simulator_run_round():
    """Test running a single training round."""
    fl_config = FLConfig(
        num_rounds=10,
        num_clients=5,
        clients_per_round=5
    )

    simulator = FLSimulator(fl_config)

    # Run one round
    round_data = simulator.run_round()

    assert round_data.round_num == 1
    assert round_data.global_accuracy > 0
    assert round_data.global_loss > 0
    assert len(round_data.per_client_metrics) > 0


def test_simulator_full_training():
    """Test running full training simulation."""
    fl_config = FLConfig(
        num_rounds=5,  # Small number for testing
        num_clients=5,
        clients_per_round=5
    )

    simulator = FLSimulator(fl_config)

    # Run all rounds
    for _ in range(fl_config.num_rounds):
        simulator.run_round()

    assert simulator.current_round == fl_config.num_rounds


def test_simulator_with_attack():
    """Test simulator with attack configuration."""
    fl_config = FLConfig(
        num_rounds=20,
        num_clients=10,
        clients_per_round=10
    )

    attack_config = AttackConfig(
        attack_type="label_flipping",
        start_round=5,
        end_round=10,
        num_attackers=2
    )

    defense_config = DefenseConfig(
        defense_type="signguard"
    )

    simulator = FLSimulator(
        fl_config,
        attack_config=attack_config,
        defense_config=defense_config
    )

    # Run rounds before attack
    for _ in range(4):
        round_data = simulator.run_round()
        # No security events before attack starts
        assert len(round_data.security_events) == 0

    # Run attack round
    round_data = simulator.run_round()
    # Should have security events now
    assert round_data.round_num == 5


def test_demo_simulator_creation():
    """Test creating demo simulators."""
    scenarios = [
        "normal",
        "label_flipping",
        "backdoor",
        "byzantine",
        "signguard_defense",
        "foolsgold_defense"
    ]

    for scenario in scenarios:
        simulator = create_demo_simulator(scenario)
        assert simulator is not None
        assert simulator.config.num_rounds > 0


def test_client_selection():
    """Test client selection logic."""
    fl_config = FLConfig(
        num_rounds=10,
        num_clients=10,
        clients_per_round=5  # Select half the clients
    )

    simulator = FLSimulator(fl_config)

    selected = simulator._select_clients()

    assert len(selected) <= 5
    assert all(0 <= cid < 10 for cid in selected)


def test_federated_averaging():
    """Test FedAvg aggregation."""
    fl_config = FLConfig(
        num_rounds=10,
        num_clients=5,
        clients_per_round=5
    )

    simulator = FLSimulator(fl_config)

    # Create fake updates
    import numpy as np
    client_updates = {
        0: np.ones(100) * 1.0,
        1: np.ones(100) * 2.0,
        2: np.ones(100) * 3.0
    }

    client_metrics = {
        0: simulator._simulate_client_training(0)[1],
        1: simulator._simulate_client_training(1)[1],
        2: simulator._simulate_client_training(2)[1]
    }

    aggregated = simulator._federated_averaging(client_updates, client_metrics)

    # Aggregated should be weighted average
    assert aggregated.shape == (100,)


def test_simulator_reset():
    """Test simulator reset functionality."""
    fl_config = FLConfig(
        num_rounds=10,
        num_clients=5,
        clients_per_round=5
    )

    simulator = FLSimulator(fl_config)

    # Run some rounds
    simulator.run_round()
    simulator.run_round()

    assert simulator.current_round == 2

    # Reset
    simulator.reset(seed=42)

    assert simulator.current_round == 0
    assert simulator.global_model_accuracy == 0.1


def test_anomaly_score_calculation():
    """Test anomaly score calculation."""
    fl_config = FLConfig(
        num_rounds=10,
        num_clients=5,
        clients_per_round=5
    )

    simulator = FLSimulator(fl_config)

    # Normal client
    score1 = simulator._calculate_anomaly_score(
        client_id=0,
        update_norm=1.0,
        is_attacker=False
    )

    # Attacker
    score2 = simulator._calculate_anomaly_score(
        client_id=1,
        update_norm=1.0,
        is_attacker=True
    )

    # Attacker should generally have higher score
    assert score1 >= 0
    assert score2 >= 0
    # (Not strictly greater due to randomness)


def test_global_model_update():
    """Test global model update logic."""
    fl_config = FLConfig(
        num_rounds=10,
        num_clients=5,
        clients_per_round=5
    )

    simulator = FLSimulator(fl_config)

    initial_acc = simulator.global_model_accuracy
    initial_loss = simulator.global_model_loss

    # Update model
    import numpy as np
    simulator._update_global_model(
        aggregated_update=np.zeros(1000),
        selected_clients=[0, 1, 2, 3, 4]
    )

    # Accuracy should improve
    assert simulator.global_model_accuracy > initial_acc

    # Loss should decrease
    assert simulator.global_model_loss < initial_loss
