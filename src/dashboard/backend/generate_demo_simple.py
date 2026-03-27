"""
Simple Demo Data Generator (No dependencies)
Creates JSON files with pre-recorded FL training scenarios.
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta


def generate_normal_training(num_rounds=50, num_clients=10):
    """Generate normal training scenario."""
    rounds = []

    for round_num in range(1, num_rounds + 1):
        # Convergence curves
        target_acc = 0.9 - 0.8 * (2.718 ** (-0.03 * round_num))
        accuracy = max(0.1, target_acc + random.gauss(0, 0.01))
        target_loss = 0.1 + 2.4 * (2.718 ** (-0.04 * round_num))
        loss = max(0.01, target_loss + random.gauss(0, 0.02))

        # Client metrics
        client_metrics = []
        for client_id in range(num_clients):
            client_acc = accuracy + random.gauss(0, 0.02)
            client_loss = loss + random.gauss(0, 0.05)

            client_metrics.append({
                "client_id": client_id,
                "accuracy": max(0, min(1, client_acc)),
                "loss": max(0, client_loss),
                "data_size": random.randint(500, 1500),
                "training_time": random.uniform(1, 5),
                "status": "active",
                "anomaly_score": random.uniform(0, 0.2),
                "update_norm": random.uniform(0.5, 2.0),
                "reputation_score": random.uniform(0.85, 1.0)
            })

        rounds.append({
            "round_num": round_num,
            "timestamp": datetime.now().isoformat(),
            "global_loss": loss,
            "global_accuracy": accuracy,
            "per_client_metrics": client_metrics,
            "loss_delta": 0.0 if round_num == 1 else loss - rounds[-1]["global_loss"],
            "accuracy_delta": 0.0 if round_num == 1 else accuracy - rounds[-1]["global_accuracy"],
            "security_events": [],
            "epsilon_spent": 0.0
        })

    return rounds


def generate_label_flipping_attack(num_rounds=50, num_clients=10):
    """Generate label flipping attack scenario."""
    rounds = []
    attackers = [8, 9]  # Client IDs that are attackers
    attack_start = 10
    attack_end = 25

    for round_num in range(1, num_rounds + 1):
        # Training with attack impact
        is_attacking = attack_start <= round_num <= attack_end

        # Attack degrades performance
        base_acc = 0.9 - 0.8 * (2.718 ** (-0.03 * round_num))
        if is_attacking:
            base_acc -= 0.05  # Attack impact

        accuracy = max(0.1, base_acc + random.gauss(0, 0.01))
        target_loss = 0.1 + 2.4 * (2.718 ** (-0.04 * round_num))
        if is_attacking:
            target_loss += 0.1
        loss = max(0.01, target_loss + random.gauss(0, 0.02))

        # Client metrics
        client_metrics = []
        security_events = []

        for client_id in range(num_clients):
            is_attacker = client_id in attackers
            is_detected = is_attacker and random.random() < 0.3

            client_acc = accuracy + random.gauss(0, 0.02)
            if is_attacker and is_attacking:
                client_acc -= 0.05  # Attacker has worse metrics

            anomaly_score = random.uniform(0, 0.2)
            if is_attacker and is_attacking:
                anomaly_score = random.uniform(0.4, 0.8)

            client_metrics.append({
                "client_id": client_id,
                "accuracy": max(0, min(1, client_acc)),
                "loss": loss + random.gauss(0, 0.05),
                "data_size": random.randint(500, 1500),
                "training_time": random.uniform(1, 5),
                "status": "anomaly" if is_detected else ("active" if not (is_attacker and is_attacking) else "idle"),
                "anomaly_score": anomaly_score,
                "update_norm": random.uniform(0.5, 2.0),
                "reputation_score": 1.0 - anomaly_score * 0.5
            })

            # Generate detection events
            if is_detected and round_num in [attack_start, attack_start + 5, attack_end]:
                security_events.append({
                    "event_id": f"lf_{round_num}_{client_id}",
                    "event_type": "attack_detected",
                    "severity": "high",
                    "message": f"Label flipping attack detected from Client {client_id}",
                    "round_num": round_num,
                    "attack_type": "label_flipping",
                    "affected_clients": [client_id],
                    "confidence": random.uniform(0.75, 0.95),
                    "timestamp": datetime.now().isoformat(),
                    "resolved": False
                })

        rounds.append({
            "round_num": round_num,
            "timestamp": datetime.now().isoformat(),
            "global_loss": loss,
            "global_accuracy": accuracy,
            "per_client_metrics": client_metrics,
            "loss_delta": 0.0 if round_num == 1 else loss - rounds[-1]["global_loss"],
            "accuracy_delta": 0.0 if round_num == 1 else accuracy - rounds[-1]["global_accuracy"],
            "security_events": security_events,
            "epsilon_spent": 0.0
        })

    return rounds


def generate_backdoor_attack(num_rounds=50, num_clients=10):
    """Generate backdoor attack scenario."""
    rounds = []
    attacker = [7]  # Single attacker
    attack_start = 15
    attack_end = 30

    for round_num in range(1, num_rounds + 1):
        is_attacking = attack_start <= round_num <= attack_end

        base_acc = 0.9 - 0.8 * (2.718 ** (-0.03 * round_num))
        accuracy = max(0.1, base_acc + random.gauss(0, 0.01))
        loss = max(0.01, 0.1 + 2.4 * (2.718 ** (-0.04 * round_num)) + random.gauss(0, 0.02))

        client_metrics = []
        security_events = []

        for client_id in range(num_clients):
            is_attacker = client_id in attacker

            anomaly_score = random.uniform(0, 0.2)
            if is_attacker and is_attacking:
                anomaly_score = random.uniform(0.3, 0.6)  # Harder to detect

            client_metrics.append({
                "client_id": client_id,
                "accuracy": max(0, min(1, accuracy + random.gauss(0, 0.02))),
                "loss": loss + random.gauss(0, 0.05),
                "data_size": random.randint(500, 1500),
                "training_time": random.uniform(1, 5),
                "status": "active",
                "anomaly_score": anomaly_score,
                "update_norm": random.uniform(0.5, 2.0),
                "reputation_score": 1.0 - anomaly_score * 0.5
            })

            # Occasional detection
            if is_attacker and is_attacking and random.random() < 0.15:
                security_events.append({
                    "event_id": f"bd_{round_num}_{client_id}",
                    "event_type": "anomaly_detected",
                    "severity": "medium",
                    "message": f"Anomalous pattern from Client {client_id}",
                    "round_num": round_num,
                    "attack_type": "backdoor",
                    "affected_clients": [client_id],
                    "confidence": random.uniform(0.5, 0.7),
                    "timestamp": datetime.now().isoformat(),
                    "resolved": False
                })

        rounds.append({
            "round_num": round_num,
            "timestamp": datetime.now().isoformat(),
            "global_loss": loss,
            "global_accuracy": accuracy,
            "per_client_metrics": client_metrics,
            "loss_delta": 0.0 if round_num == 1 else loss - rounds[-1]["global_loss"],
            "accuracy_delta": 0.0 if round_num == 1 else accuracy - rounds[-1]["global_accuracy"],
            "security_events": security_events,
            "epsilon_spent": 0.0
        })

    return rounds


def generate_byzantine_attack(num_rounds=50, num_clients=10):
    """Generate Byzantine attack scenario."""
    rounds = []
    attackers = [6, 7, 8]  # 3 attackers
    attack_start = 10
    attack_end = 40

    for round_num in range(1, num_rounds + 1):
        is_attacking = attack_start <= round_num <= attack_end

        # Byzantine severely impacts training
        base_acc = 0.9 - 0.8 * (2.718 ** (-0.03 * round_num))
        if is_attacking:
            base_acc -= 0.1  # Significant impact

        accuracy = max(0.1, base_acc + random.gauss(0, 0.01))
        loss = max(0.01, 0.1 + 2.4 * (2.718 ** (-0.04 * round_num)) + random.gauss(0, 0.02))

        client_metrics = []
        security_events = []

        for client_id in range(num_clients):
            is_attacker = client_id in attackers

            anomaly_score = random.uniform(0, 0.2)
            if is_attacker and is_attacking:
                anomaly_score = random.uniform(0.6, 0.95)

            client_metrics.append({
                "client_id": client_id,
                "accuracy": max(0, min(1, accuracy + random.gauss(0, 0.02))),
                "loss": loss + random.gauss(0, 0.05),
                "data_size": random.randint(500, 1500),
                "training_time": random.uniform(1, 5),
                "status": "anomaly" if anomaly_score > 0.7 else "active",
                "anomaly_score": anomaly_score,
                "update_norm": random.uniform(0.5, 2.0),
                "reputation_score": 1.0 - anomaly_score * 0.5
            })

            # Multiple detections
            if is_attacker and is_attacking and random.random() < 0.5:
                security_events.append({
                    "event_id": f"bz_{round_num}_{client_id}",
                    "event_type": "attack_detected",
                    "severity": "high",
                    "message": f"Byzantine attack from Client {client_id}",
                    "round_num": round_num,
                    "attack_type": "byzantine",
                    "affected_clients": [client_id],
                    "confidence": random.uniform(0.7, 0.95),
                    "timestamp": datetime.now().isoformat(),
                    "resolved": False
                })

        rounds.append({
            "round_num": round_num,
            "timestamp": datetime.now().isoformat(),
            "global_loss": loss,
            "global_accuracy": accuracy,
            "per_client_metrics": client_metrics,
            "loss_delta": 0.0 if round_num == 1 else loss - rounds[-1]["global_loss"],
            "accuracy_delta": 0.0 if round_num == 1 else accuracy - rounds[-1]["global_accuracy"],
            "security_events": security_events,
            "epsilon_spent": 0.0
        })

    return rounds


def generate_signguard_defense(num_rounds=50, num_clients=10):
    """Generate SignGuard defense scenario."""
    rounds = []
    attackers = [8, 9]
    attack_start = 10

    for round_num in range(1, num_rounds + 1):
        is_attacking = round_num >= attack_start

        base_acc = 0.9 - 0.8 * (2.718 ** (-0.03 * round_num))
        # SignGuard mitigates attack well
        accuracy = max(0.1, base_acc + random.gauss(0, 0.01))
        loss = max(0.01, 0.1 + 2.4 * (2.718 ** (-0.04 * round_num)) + random.gauss(0, 0.02))

        client_metrics = []
        security_events = []

        for client_id in range(num_clients):
            is_attacker = client_id in attackers

            anomaly_score = random.uniform(0, 0.2)
            if is_attacker and is_attacking:
                anomaly_score = random.uniform(0.5, 0.8)

            # Reputation drops for attackers
            reputation = 1.0
            if is_attacker and is_attacking:
                reputation = max(0.2, 1.0 - (round_num - attack_start) * 0.05)

            client_metrics.append({
                "client_id": client_id,
                "accuracy": max(0, min(1, accuracy + random.gauss(0, 0.02))),
                "loss": loss + random.gauss(0, 0.05),
                "data_size": random.randint(500, 1500),
                "training_time": random.uniform(1, 5),
                "status": "active",
                "anomaly_score": anomaly_score,
                "update_norm": random.uniform(0.5, 2.0),
                "reputation_score": reputation
            })

            # SignGuard defense events
            if is_attacker and is_attacking and round_num > attack_start:
                security_events.append({
                    "event_id": f"sg_{round_num}_{client_id}",
                    "event_type": "defense_activated",
                    "severity": "medium",
                    "message": f"SignGuard downweighted Client {client_id} (rep: {reputation:.2f})",
                    "round_num": round_num,
                    "attack_type": None,
                    "affected_clients": [client_id],
                    "confidence": anomaly_score,
                    "timestamp": datetime.now().isoformat(),
                    "resolved": True
                })

        rounds.append({
            "round_num": round_num,
            "timestamp": datetime.now().isoformat(),
            "global_loss": loss,
            "global_accuracy": accuracy,
            "per_client_metrics": client_metrics,
            "loss_delta": 0.0 if round_num == 1 else loss - rounds[-1]["global_loss"],
            "accuracy_delta": 0.0 if round_num == 1 else accuracy - rounds[-1]["global_accuracy"],
            "security_events": security_events,
            "epsilon_spent": 0.0
        })

    return rounds


def generate_foolsgold_defense(num_rounds=50, num_clients=10):
    """Generate FoolsGold defense scenario."""
    rounds = []
    attackers = [6, 7, 8]
    attack_start = 10

    for round_num in range(1, num_rounds + 1):
        is_attacking = round_num >= attack_start

        base_acc = 0.9 - 0.8 * (2.718 ** (-0.03 * round_num))
        accuracy = max(0.1, base_acc + random.gauss(0, 0.01))
        loss = max(0.01, 0.1 + 2.4 * (2.718 ** (-0.04 * round_num)) + random.gauss(0, 0.02))

        client_metrics = []
        security_events = []

        for client_id in range(num_clients):
            is_attacker = client_id in attackers

            # FoolsGold detects colluding attackers by similarity
            anomaly_score = random.uniform(0, 0.2)
            reputation = 1.0

            if is_attacker and is_attacking:
                # Attackers have similar updates (high similarity)
                reputation = 0.3 + random.uniform(0, 0.2)

            client_metrics.append({
                "client_id": client_id,
                "accuracy": max(0, min(1, accuracy + random.gauss(0, 0.02))),
                "loss": loss + random.gauss(0, 0.05),
                "data_size": random.randint(500, 1500),
                "training_time": random.uniform(1, 5),
                "status": "active",
                "anomaly_score": anomaly_score,
                "update_norm": random.uniform(0.5, 2.0),
                "reputation_score": reputation
            })

            # FoolsGold events
            if is_attacker and is_attacking and random.random() < 0.4:
                security_events.append({
                    "event_id": f"fg_{round_num}_{client_id}",
                    "event_type": "defense_activated",
                    "severity": "low",
                    "message": f"FoolsGold downweighted Client {client_id}",
                    "round_num": round_num,
                    "attack_type": None,
                    "affected_clients": [client_id],
                    "confidence": 1.0 - reputation,
                    "timestamp": datetime.now().isoformat(),
                    "resolved": True
                })

        rounds.append({
            "round_num": round_num,
            "timestamp": datetime.now().isoformat(),
            "global_loss": loss,
            "global_accuracy": accuracy,
            "per_client_metrics": client_metrics,
            "loss_delta": 0.0 if round_num == 1 else loss - rounds[-1]["global_loss"],
            "accuracy_delta": 0.0 if round_num == 1 else accuracy - rounds[-1]["global_accuracy"],
            "security_events": security_events,
            "epsilon_spent": 0.0
        })

    return rounds


def save_scenario(scenario_name, rounds, output_dir):
    """Save scenario to JSON file."""
    output = {
        "scenario": scenario_name,
        "generated_at": datetime.now().isoformat(),
        "rounds": rounds
    }

    filename = f"{scenario_name}.json"
    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(rounds)} rounds to {filename}")


def main():
    """Generate all demo scenarios."""
    output_dir = Path(__file__).parent.parent / "data" / "demo_scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Generating FL Security Dashboard Demo Data")
    print("=" * 50)

    scenarios = [
        ("normal_training", generate_normal_training()),
        ("label_flipping", generate_label_flipping_attack()),
        ("backdoor", generate_backdoor_attack()),
        ("byzantine", generate_byzantine_attack()),
        ("signguard_defense", generate_signguard_defense()),
        ("foolsgold_defense", generate_foolsgold_defense())
    ]

    for name, rounds in scenarios:
        print(f"Generating scenario: {name}")
        save_scenario(name, rounds, output_dir)

    print("=" * 50)
    print(f"Done! Generated {len(scenarios)} scenarios")
    print(f"Output directory: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
