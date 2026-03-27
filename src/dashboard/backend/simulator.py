"""
Federated Learning Training Simulator
Simulates FL training with attacks and defenses for dashboard demonstrations.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from core.data_models import (
    TrainingRound,
    ClientMetric,
    SecurityEvent,
    FLConfig,
    AttackConfig,
    DefenseConfig
)
from core.attack_engine import AttackEngine
from core.defense_engine import DefenseEngine


class FLSimulator:
    """
    Simulates Federated Learning training for demonstration purposes.
    Generates realistic training data with configurable attacks and defenses.
    """

    def __init__(
        self,
        fl_config: FLConfig,
        attack_config: Optional[AttackConfig] = None,
        defense_config: Optional[DefenseConfig] = None,
        seed: int = 42
    ):
        """
        Initialize FL simulator.

        Args:
            fl_config: FL training configuration
            attack_config: Attack configuration (None = no attack)
            defense_config: Defense configuration
            seed: Random seed
        """
        self.config = fl_config
        self.attack_config = attack_config
        self.defense_config = defense_config or DefenseConfig(defense_type="none")
        self.seed = seed

        self.rng = np.random.RandomState(seed)

        # Initialize attack and defense engines
        self.attack_engine = AttackEngine(attack_config, seed) if attack_config else None
        self.defense_engine = DefenseEngine(defense_config, fl_config.num_clients)

        # Training state
        self.current_round = 0
        self.global_model_accuracy = 0.1  # Start with random accuracy
        self.global_model_loss = 2.5  # Start with high loss

        # Per-client state
        self.client_states = self._initialize_clients()

    def _initialize_clients(self) -> Dict[int, Dict]:
        """Initialize client-specific states."""
        clients = {}
        for i in range(self.config.num_clients):
            clients[i] = {
                "data_size": self.rng.randint(500, 1500),
                "base_accuracy": self.rng.uniform(0.4, 0.6),  # Individual client performance
                "reputation": 1.0,
                "update_norm_base": self.rng.uniform(0.8, 1.2)
            }
        return clients

    def run_round(self) -> TrainingRound:
        """
        Simulate one training round.

        Returns:
            TrainingRound with all metrics
        """
        self.current_round += 1

        # Select clients for this round
        selected_clients = self._select_clients()

        # Simulate client training
        client_updates = {}
        client_metrics = {}

        for client_id in selected_clients:
            update_vector, metric = self._simulate_client_training(client_id)
            client_updates[client_id] = update_vector
            client_metrics[client_id] = metric

        # Apply attack if configured
        security_events = []
        if self.attack_engine and self.attack_engine.should_attack_this_round(self.current_round):
            client_updates, attack_events = self.attack_engine.apply_attack(
                self.current_round,
                client_updates,
                client_metrics
            )
            security_events.extend(attack_events)

        # Apply defense
        if self.defense_config.defense_type != "none":
            client_updates, defense_events = self.defense_engine.apply_defense(
                self.current_round,
                client_updates,
                client_metrics
            )
            security_events.extend(defense_events)

        # Aggregate updates (FedAvg)
        aggregated_update = self._federated_averaging(client_updates, client_metrics)

        # Update global model
        self._update_global_model(aggregated_update, selected_clients)

        # Calculate round delta
        loss_delta = 0.0 if self.current_round == 1 else self.global_model_loss - self._prev_loss
        accuracy_delta = 0.0 if self.current_round == 1 else self.global_model_accuracy - self._prev_accuracy

        # Store for next round
        self._prev_loss = self.global_model_loss
        self._prev_accuracy = self.global_model_accuracy

        # Calculate DP epsilon if enabled
        epsilon_spent = 0.1 if self.config.use_dp else 0.0

        # Create training round
        training_round = TrainingRound(
            round_num=self.current_round,
            timestamp=datetime.now(),
            global_loss=self.global_model_loss,
            global_accuracy=self.global_model_accuracy,
            per_client_metrics=list(client_metrics.values()),
            loss_delta=loss_delta,
            accuracy_delta=accuracy_delta,
            security_events=security_events,
            epsilon_spent=epsilon_spent
        )

        return training_round

    def _select_clients(self) -> List[int]:
        """Select clients for this round."""
        num_selected = min(self.config.clients_per_round, self.config.num_clients)

        # Simple random selection (could implement more sophisticated strategies)
        selected = self.rng.choice(
            self.config.num_clients,
            num_selected,
            replace=False
        )

        # Simulate client drops
        if self.config.drop_rate > 0:
            keep_mask = self.rng.rand(num_selected) > self.config.drop_rate
            selected = selected[keep_mask]

        return selected.tolist()

    def _simulate_client_training(self, client_id: int) -> Tuple[np.ndarray, ClientMetric]:
        """
        Simulate local training for one client.

        Returns:
            Tuple of (update_vector, client_metric)
        """
        client_state = self.client_states[client_id]

        # Simulate training time
        training_time = self.rng.uniform(1.0, 5.0) * (self.config.local_epochs / 5)

        # Calculate client's model improvement
        # Clients improve based on their base capability + global progress
        round_factor = 1.0 - np.exp(-0.05 * self.current_round)  # Convergence curve
        client_accuracy = client_state["base_accuracy"] + 0.35 * round_factor
        client_accuracy += self.rng.randn() * 0.02  # Add noise

        # Accuracy loss due to attacks if this client is an attacker
        is_attacker = (
            self.attack_engine and
            self.attack_engine.should_attack_this_round(self.current_round) and
            client_id in self.attack_engine._get_attacker_ids(list(range(self.config.num_clients)))
        )

        if is_attacker:
            # Attackers might have slightly different behavior
            client_accuracy += self.rng.uniform(-0.1, 0.05)

        client_accuracy = np.clip(client_accuracy, 0, 1)

        # Loss is inverse of accuracy
        client_loss = -np.log(client_accuracy + 0.01) + self.rng.randn() * 0.05
        client_loss = max(0.01, client_loss)

        # Generate update vector (simplified - just a random vector)
        model_size = 1000  # Simulated model size
        update_vector = self.rng.randn(model_size) * 0.1 * client_state["update_norm_base"]

        # Update norm for anomaly detection
        update_norm = np.linalg.norm(update_vector)

        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(client_id, update_norm, is_attacker)

        # Update reputation
        self.defense_engine.reputation_scores[client_id] = (
            self.defense_engine.reputation_scores.get(client_id, 1.0) *
            (1.0 - anomaly_score * 0.1)
        )

        # Determine status
        if anomaly_score > 0.7:
            status = "anomaly"
        elif self.rng.rand() < 0.05:  # Random drops
            status = "idle"
        else:
            status = "active"

        # Create client metric
        metric = ClientMetric(
            client_id=client_id,
            accuracy=client_accuracy,
            loss=client_loss,
            data_size=client_state["data_size"],
            training_time=training_time,
            status=status,
            anomaly_score=anomaly_score,
            update_norm=update_norm,
            reputation_score=self.defense_engine.reputation_scores.get(client_id, 1.0)
        )

        return update_vector, metric

    def _calculate_anomaly_score(self, client_id: int, update_norm: float, is_attacker: bool) -> float:
        """Calculate anomaly score for a client."""
        base_score = self.rng.uniform(0, 0.2)  # Most clients are normal

        if is_attacker:
            # Attackers have higher anomaly scores (but not always detected)
            base_score = 0.4 + self.rng.uniform(0, 0.6)

        # Occasionally flag normal clients (false positives)
        if self.rng.rand() < 0.05:
            base_score = max(base_score, 0.5)

        return min(1.0, base_score)

    def _federated_averaging(
        self,
        client_updates: Dict[int, np.ndarray],
        client_metrics: Dict[int, ClientMetric]
    ) -> np.ndarray:
        """
        Perform FedAvg aggregation.

        Returns:
            Aggregated update vector
        """
        if not client_updates:
            return np.zeros(1000)

        # Weight by data size
        total_data = sum(m.data_size for m in client_metrics.values())
        weights = {cid: m.data_size / total_data for cid, m in client_metrics.items()}

        # Weighted average
        aggregated = sum(
            weights[cid] * client_updates[cid]
            for cid in client_updates.keys()
        )

        return aggregated

    def _update_global_model(self, aggregated_update: np.ndarray, selected_clients: List[int]) -> None:
        """Update global model based on aggregated update."""
        # Simulate model improvement
        # More clients = better improvement
        improvement_factor = len(selected_clients) / self.config.clients_per_round

        # Learning rate effect
        lr_effect = self.config.learning_rate * 100

        # Update accuracy (logistic-like curve)
        target_accuracy = 0.9 - 0.8 * np.exp(-0.03 * self.current_round)
        accuracy_step = (target_accuracy - self.global_model_accuracy) * 0.1 * improvement_factor * lr_effect
        self.global_model_accuracy += accuracy_step
        self.global_model_accuracy = np.clip(self.global_model_accuracy, 0, 1)

        # Update loss (decreases over time)
        target_loss = 0.1 + 2.4 * np.exp(-0.04 * self.current_round)
        loss_step = (target_loss - self.global_model_loss) * 0.1 * improvement_factor * lr_effect
        self.global_model_loss += loss_step
        self.global_model_loss = max(0.01, self.global_model_loss)

    def run_full_training(self, callback=None) -> List[TrainingRound]:
        """
        Run complete training simulation.

        Args:
            callback: Optional function called after each round

        Returns:
            List of all training rounds
        """
        rounds = []

        for _ in range(self.config.num_rounds):
            round_data = self.run_round()
            rounds.append(round_data)

            if callback:
                callback(round_data)

            # Small delay to simulate real training
            time.sleep(0.01)

        return rounds

    def get_client_reputations(self) -> Dict[int, float]:
        """Get current client reputations."""
        return self.defense_engine.get_reputation_scores()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset simulator state."""
        if seed is not None:
            self.seed = seed

        self.rng = np.random.RandomState(self.seed)
        self.current_round = 0
        self.global_model_accuracy = 0.1
        self.global_model_loss = 2.5
        self.client_states = self._initialize_clients()

        # Reset engines
        if self.attack_engine:
            self.attack_engine = AttackEngine(self.attack_config, self.seed)

        self.defense_engine = DefenseEngine(self.defense_config, self.config.num_clients)


def create_demo_simulator(scenario: str = "normal") -> FLSimulator:
    """
    Create a simulator with pre-configured scenario.

    Args:
        scenario: Scenario name (normal, label_flipping, backdoor, byzantine, etc.)

    Returns:
        Configured FLSimulator
    """
    # Base FL config
    fl_config = FLConfig(
        num_rounds=50,
        num_clients=10,
        clients_per_round=10,
        learning_rate=0.01,
        local_epochs=5
    )

    attack_config = None
    defense_config = None

    if scenario == "normal":
        defense_config = DefenseConfig(defense_type="none")

    elif scenario == "label_flipping":
        attack_config = AttackConfig(
            attack_type="label_flipping",
            start_round=10,
            end_round=25,
            num_attackers=2
        )
        defense_config = DefenseConfig(defense_type="signguard")

    elif scenario == "backdoor":
        attack_config = AttackConfig(
            attack_type="backdoor",
            start_round=15,
            end_round=30,
            num_attackers=1
        )
        defense_config = DefenseConfig(defense_type="krum")

    elif scenario == "byzantine":
        attack_config = AttackConfig(
            attack_type="byzantine",
            start_round=10,
            end_round=40,
            num_attackers=3,
            byzantine_type="sign_flip"
        )
        defense_config = DefenseConfig(defense_type="foolsgold")

    elif scenario == "signguard_defense":
        attack_config = AttackConfig(
            attack_type="label_flipping",
            start_round=10,
            num_attackers=2
        )
        defense_config = DefenseConfig(
            defense_type="signguard",
            anomaly_threshold=0.5,
            action_on_detection="downweight"
        )

    elif scenario == "foolsgold_defense":
        attack_config = AttackConfig(
            attack_type="byzantine",
            start_round=10,
            num_attackers=3
        )
        defense_config = DefenseConfig(
            defense_type="foolsgold"
        )

    else:
        # Default to normal
        defense_config = DefenseConfig(defense_type="none")

    return FLSimulator(fl_config, attack_config, defense_config)
