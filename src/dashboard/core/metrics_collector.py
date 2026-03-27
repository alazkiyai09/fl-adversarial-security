"""
Metrics collection and aggregation for the FL dashboard.
Handles real-time metric collection and efficient aggregation.
"""

import numpy as np
from typing import List, Dict, Optional, DefaultDict
from collections import defaultdict
from datetime import datetime
from .data_models import (
    TrainingRound,
    ClientMetric,
    SecurityEvent,
    ExperimentResult,
    PrivacyBudget
)


class MetricsCollector:
    """
    Collects and aggregates metrics from FL training.
    Maintains training history and provides efficient queries.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum number of rounds to keep in memory
        """
        self.max_history = max_history

        # Training history
        self.training_history: List[TrainingRound] = []

        # Security event log
        self.security_events: List[SecurityEvent] = []

        # Client-specific metrics history
        self.client_history: DefaultDict[int, List[ClientMetric]] = defaultdict(list)

        # Privacy tracking
        self.privacy_budget: Optional[PrivacyBudget] = None

        # Current round
        self.current_round: int = 0

    def add_round(self, round_data: TrainingRound) -> None:
        """
        Add a complete training round to history.

        Args:
            round_data: Training round data
        """
        self.training_history.append(round_data)
        self.current_round = round_data.round_num

        # Add client metrics to history
        for client_metric in round_data.per_client_metrics:
            self.client_history[client_metric.client_id].append(client_metric)

        # Add security events
        self.security_events.extend(round_data.security_events)

        # Prune old history if needed
        if len(self.training_history) > self.max_history:
            removed = self.training_history.pop(0)
            # Remove associated client metrics
            for client_metric in removed.per_client_metrics:
                if client_metric.client_id in self.client_history:
                    if self.client_history[client_metric.client_id]:
                        self.client_history[client_metric.client_id].pop(0)

    def set_privacy_budget(self, budget: PrivacyBudget) -> None:
        """Set/update privacy budget tracking."""
        self.privacy_budget = budget

    def get_training_history(
        self,
        start_round: Optional[int] = None,
        end_round: Optional[int] = None
    ) -> List[TrainingRound]:
        """
        Get training history for a range of rounds.

        Args:
            start_round: Starting round (None = from beginning)
            end_round: Ending round (None = to current)

        Returns:
            List of training rounds
        """
        history = self.training_history

        if start_round is not None:
            history = [r for r in history if r.round_num >= start_round]
        if end_round is not None:
            history = [r for r in history if r.round_num <= end_round]

        return history

    def get_recent_rounds(self, n: int = 10) -> List[TrainingRound]:
        """Get the most recent n rounds."""
        return self.training_history[-n:] if n > 0 else []

    def get_loss_curve(self) -> List[tuple]:
        """
        Get loss values across rounds.

        Returns:
            List of (round_num, loss) tuples
        """
        return [(r.round_num, r.global_loss) for r in self.training_history]

    def get_accuracy_curve(self) -> List[tuple]:
        """
        Get accuracy values across rounds.

        Returns:
            List of (round_num, accuracy) tuples
        """
        return [(r.round_num, r.global_accuracy) for r in self.training_history]

    def get_client_metrics(
        self,
        client_id: Optional[int] = None,
        n_recent: int = 1
    ) -> Dict[int, List[ClientMetric]]:
        """
        Get metrics for specific client(s).

        Args:
            client_id: Specific client ID (None = all clients)
            n_recent: Number of recent metrics to return

        Returns:
            Dictionary mapping client_id to list of metrics
        """
        if client_id is not None:
            return {client_id: self.client_history[client_id][-n_recent:]}

        return {
            cid: metrics[-n_recent:]
            for cid, metrics in self.client_history.items()
        }

    def get_client_reputations(self) -> Dict[int, float]:
        """Get current reputation scores for all clients."""
        reputations = {}
        for client_id, metrics_list in self.client_history.items():
            if metrics_list:
                # Get most recent reputation
                reputations[client_id] = metrics_list[-1].reputation_score
            else:
                reputations[client_id] = 1.0  # Default
        return reputations

    def get_anomaly_scores(self) -> Dict[int, float]:
        """Get current anomaly scores for all clients."""
        scores = {}
        for client_id, metrics_list in self.client_history.items():
            if metrics_list:
                scores[client_id] = metrics_list[-1].anomaly_score
            else:
                scores[client_id] = 0.0
        return scores

    def get_client_status_counts(self) -> Dict[str, int]:
        """
        Get count of clients by status.

        Returns:
            Dictionary with status counts
        """
        if not self.training_history:
            return {"active": 0, "idle": 0, "dropped": 0, "anomaly": 0}

        latest_round = self.training_history[-1]
        status_counts = {"active": 0, "idle": 0, "dropped": 0, "anomaly": 0}

        for metric in latest_round.per_client_metrics:
            status_counts[metric.status] += 1

        return status_counts

    def get_security_events(
        self,
        n_recent: int = 20,
        severity: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[SecurityEvent]:
        """
        Get recent security events with optional filtering.

        Args:
            n_recent: Number of recent events
            severity: Filter by severity (None = all)
            event_type: Filter by event type (None = all)

        Returns:
            List of security events
        """
        events = self.security_events[-n_recent:]

        if severity is not None:
            events = [e for e in events if e.severity == severity]

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        return events

    def get_attack_statistics(self) -> Dict:
        """Get statistics about detected attacks."""
        events = self.security_events

        return {
            "total_events": len(events),
            "attacks_detected": len([e for e in events if e.event_type == "attack_detected"]),
            "defenses_activated": len([e for e in events if e.event_type == "defense_activated"]),
            "by_severity": {
                "critical": len([e for e in events if e.severity == "critical"]),
                "high": len([e for e in events if e.severity == "high"]),
                "medium": len([e for e in events if e.severity == "medium"]),
                "low": len([e for e in events if e.severity == "low"])
            },
            "by_attack_type": self._count_by_attack_type(events)
        }

    def _count_by_attack_type(self, events: List[SecurityEvent]) -> Dict[str, int]:
        """Count events by attack type."""
        attack_types = {}
        for event in events:
            if event.attack_type:
                attack_types[event.attack_type] = attack_types.get(event.attack_type, 0) + 1
        return attack_types

    def get_privacy_status(self) -> Optional[Dict]:
        """Get current privacy budget status."""
        if self.privacy_budget is None:
            return None

        return {
            "epsilon_total": self.privacy_budget.epsilon_total,
            "epsilon_spent": self.privacy_budget.epsilon_spent,
            "epsilon_remaining": self.privacy_budget.epsilon_remaining,
            "percentage_used": (
                self.privacy_budget.epsilon_spent / self.privacy_budget.epsilon_total * 100
                if self.privacy_budget.epsilon_total > 0 else 0
            ),
            "secure_aggregation": self.privacy_budget.secure_aggregation,
            "encryption_method": self.privacy_budget.encryption_method
        }

    def get_convergence_metrics(self) -> Dict:
        """
        Get metrics related to training convergence.

        Returns:
            Dictionary with convergence metrics
        """
        if len(self.training_history) < 2:
            return {
                "is_converged": False,
                "convergence_rate": 0.0,
                "recent_loss_delta": 0.0
            }

        recent_rounds = self.training_history[-5:]
        loss_deltas = [r.loss_delta for r in recent_rounds if r.loss_delta is not None]
        avg_loss_delta = np.mean(loss_deltas) if loss_deltas else 0.0

        # Simple convergence check: loss change < threshold
        is_converged = abs(avg_loss_delta) < 0.001

        return {
            "is_converged": is_converged,
            "convergence_rate": abs(avg_loss_delta),
            "recent_loss_delta": avg_loss_delta,
            "rounds_trained": len(self.training_history)
        }

    def get_experiment_summary(self) -> Dict:
        """Get summary of the current/last experiment."""
        if not self.training_history:
            return {
                "status": "not_started",
                "rounds_completed": 0,
                "final_accuracy": 0.0,
                "final_loss": 0.0
            }

        latest = self.training_history[-1]

        return {
            "status": "completed" if self.training_history else "running",
            "rounds_completed": len(self.training_history),
            "final_accuracy": latest.global_accuracy,
            "final_loss": latest.global_loss,
            "total_events": len(self.security_events),
            "clients_participating": len(latest.per_client_metrics)
        }

    def export_experiment_result(self, experiment_id: str, name: str) -> ExperimentResult:
        """
        Export current metrics as an ExperimentResult.

        Args:
            experiment_id: Unique experiment identifier
            name: Experiment name

        Returns:
            ExperimentResult object
        """
        if not self.training_history:
            raise ValueError("No training data to export")

        latest = self.training_history[-1]

        return ExperimentResult(
            experiment_id=experiment_id,
            name=name,
            config=None,  # Would be set by caller
            training_history=self.training_history.copy(),
            final_accuracy=latest.global_accuracy,
            final_loss=latest.global_loss,
            total_attacks_detected=len([
                e for e in self.security_events if e.event_type == "attack_detected"
            ]),
            total_defense_activations=len([
                e for e in self.security_events if e.event_type == "defense_activated"
            ]),
            final_epsilon_spent=(
                self.privacy_budget.epsilon_spent if self.privacy_budget else 0.0
            ),
            rounds_completed=len(self.training_history),
            status="completed"
        )

    def reset(self) -> None:
        """Reset all metrics (for new experiment)."""
        self.training_history.clear()
        self.security_events.clear()
        self.client_history.clear()
        self.current_round = 0
        self.privacy_budget = None

    def get_summary_statistics(self) -> Dict:
        """Get high-level summary for dashboard overview."""
        if not self.training_history:
            return {
                "total_rounds": 0,
                "current_round": 0,
                "active_clients": 0,
                "total_events": 0,
                "threat_level": "none"
            }

        latest = self.training_history[-1]
        recent_events = self.security_events[-10:] if self.security_events else []

        # Determine threat level
        critical_events = len([e for e in recent_events if e.severity == "critical"])
        high_events = len([e for e in recent_events if e.severity == "high"])

        if critical_events > 0:
            threat_level = "critical"
        elif high_events > 2:
            threat_level = "high"
        elif high_events > 0:
            threat_level = "medium"
        else:
            threat_level = "low"

        return {
            "total_rounds": len(self.training_history),
            "current_round": self.current_round,
            "active_clients": len([
                m for m in latest.per_client_metrics if m.status == "active"
            ]),
            "total_events": len(self.security_events),
            "threat_level": threat_level,
            "global_accuracy": latest.global_accuracy,
            "global_loss": latest.global_loss
        }
