"""
Demo Data Generator
Creates JSON files with pre-recorded FL training scenarios for dashboard demonstrations.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

from backend.simulator import create_demo_simulator
from core.data_models import TrainingRound


def serialize_training_round(round_data: TrainingRound) -> Dict[str, Any]:
    """Convert TrainingRound to JSON-serializable dict."""
    return {
        "round_num": round_data.round_num,
        "timestamp": round_data.timestamp.isoformat(),
        "global_loss": round_data.global_loss,
        "global_accuracy": round_data.global_accuracy,
        "per_client_metrics": [
            {
                "client_id": m.client_id,
                "accuracy": m.accuracy,
                "loss": m.loss,
                "data_size": m.data_size,
                "training_time": m.training_time,
                "status": m.status,
                "anomaly_score": m.anomaly_score,
                "update_norm": m.update_norm,
                "reputation_score": m.reputation_score
            }
            for m in round_data.per_client_metrics
        ],
        "loss_delta": round_data.loss_delta,
        "accuracy_delta": round_data.accuracy_delta,
        "security_events": [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "severity": e.severity,
                "message": e.message,
                "round_num": e.round_num,
                "attack_type": e.attack_type,
                "affected_clients": e.affected_clients,
                "confidence": e.confidence,
                "timestamp": e.timestamp.isoformat(),
                "resolved": e.resolved
            }
            for e in round_data.security_events
        ],
        "epsilon_spent": round_data.epsilon_spent
    }


def generate_scenario(scenario_name: str, output_dir: Path) -> None:
    """
    Generate and save a demo scenario.

    Args:
        scenario_name: Name of scenario
        output_dir: Directory to save output
    """
    print(f"Generating scenario: {scenario_name}")

    simulator = create_demo_simulator(scenario_name)
    rounds_data = []

    # Run simulation
    for _ in range(simulator.config.num_rounds):
        round_data = simulator.run_round()
        rounds_data.append(serialize_training_round(round_data))

    # Prepare output
    output = {
        "scenario": scenario_name,
        "generated_at": datetime.now().isoformat(),
        "config": {
            "fl": simulator.config.model_dump(),
            "attack": simulator.attack_config.model_dump() if simulator.attack_config else None,
            "defense": simulator.defense_config.model_dump()
        },
        "rounds": rounds_data
    }

    # Save to file
    filename = f"{scenario_name}.json"
    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(rounds_data)} rounds to {filename}")


def generate_all_scenarios() -> None:
    """Generate all demo scenarios."""
    output_dir = Path(__file__).parent.parent / "data" / "demo_scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        "normal",
        "label_flipping",
        "backdoor",
        "byzantine",
        "signguard_defense",
        "foolsgold_defense"
    ]

    print("=" * 50)
    print("Generating FL Security Dashboard Demo Data")
    print("=" * 50)

    for scenario in scenarios:
        generate_scenario(scenario, output_dir)

    print("=" * 50)
    print(f"Done! Generated {len(scenarios)} scenarios")
    print(f"Output directory: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    generate_all_scenarios()
