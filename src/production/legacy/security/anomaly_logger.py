"""Anomaly logging for federated learning security.

Logs suspicious patterns, attacks, and security events.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from loguru import logger


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies."""
    POISONING_ATTACK = "poisoning_attack"
    BACKDOOR_ATTACK = "backdoor_attack"
    LABEL_FLIPPING = "label_flipping"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    UPDATE_OUTLIER = "update_outlier"
    GRADIENT_ANOMALY = "gradient_anomaly"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    COMMUNICATION_FAILURE = "communication_failure"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class AnomalyEvent:
    """An anomaly event record."""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    client_id: Optional[int] = None
    round_num: Optional[int] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "client_id": self.client_id,
            "round_num": self.round_num,
            "message": self.message,
            "details": self.details,
            "confidence": self.confidence,
            "resolved": self.resolved,
        }


class AnomalyLogger:
    """
    Logger for security anomalies and events.

    Features:
    - Structured logging to file and console
    - Event history and querying
    - Aggregation and statistics
    - Alert triggering
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        console_output: bool = True,
        retention_days: int = 30,
    ):
        """
        Initialize anomaly logger.

        Args:
            log_file: Path to log file (JSON Lines format)
            console_output: Whether to output to console
            retention_days: Retention period for logs
        """
        self.log_file = Path(log_file) if log_file else None
        self.console_output = console_output
        self.retention_days = retention_days

        # Event history
        self.events: List[AnomalyEvent] = []

        # Statistics
        self.stats: Dict[str, int] = {}

        # Create log file directory if needed
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        message: str,
        client_id: Optional[int] = None,
        round_num: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
    ) -> AnomalyEvent:
        """
        Log an anomaly event.

        Args:
            anomaly_type: Type of anomaly
            severity: Severity level
            message: Human-readable message
            client_id: Client ID (if applicable)
            round_num: Round number (if applicable)
            details: Additional details
            confidence: Detection confidence [0, 1]

        Returns:
            AnomalyEvent that was logged
        """
        event = AnomalyEvent(
            timestamp=datetime.now(),
            anomaly_type=anomaly_type,
            severity=severity,
            client_id=client_id,
            round_num=round_num,
            message=message,
            details=details or {},
            confidence=confidence,
        )

        # Add to history
        self.events.append(event)

        # Update statistics
        self._update_stats(event)

        # Console output
        if self.console_output:
            self._log_to_console(event)

        # File output
        if self.log_file:
            self._log_to_file(event)

        return event

    def _update_stats(self, event: AnomalyEvent) -> None:
        """Update anomaly statistics."""
        key = f"{event.anomaly_type.value}_{event.severity.value}"
        self.stats[key] = self.stats.get(key, 0) + 1

    def _log_to_console(self, event: AnomalyEvent) -> None:
        """Log event to console."""
        # Format message
        client_str = f"Client {event.client_id}" if event.client_id is not None else "Server"
        round_str = f"Round {event.round_num}" if event.round_num is not None else ""

        parts = [
            f"[SECURITY]",
            f"{event.severity.value.upper()}",
            event.anomaly_type.value,
            client_str,
            round_str,
            event.message,
        ]

        # Filter out empty parts
        message = " | ".join(p for p in parts if p)

        # Log with appropriate level
        if event.severity == AnomalySeverity.CRITICAL:
            logger.error(message)
        elif event.severity == AnomalySeverity.HIGH:
            logger.warning(message)
        elif event.severity == AnomalySeverity.MEDIUM:
            logger.info(message)
        else:
            logger.debug(message)

        # Log details if available
        if event.details:
            logger.debug(f"  Details: {json.dumps(event.details, indent=2, default=str)}")

    def _log_to_file(self, event: AnomalyEvent) -> None:
        """Log event to file (JSON Lines format)."""
        try:
            with open(self.log_file, "a") as f:
                json.dump(event.to_dict(), f, default=str)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def get_events(
        self,
        anomaly_type: Optional[AnomalyType] = None,
        severity: Optional[AnomalySeverity] = None,
        client_id: Optional[int] = None,
        round_num: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[AnomalyEvent]:
        """
        Query events with filters.

        Args:
            anomaly_type: Filter by anomaly type
            severity: Filter by severity
            client_id: Filter by client ID
            round_num: Filter by round number
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        filtered = self.events

        if anomaly_type is not None:
            filtered = [e for e in filtered if e.anomaly_type == anomaly_type]

        if severity is not None:
            filtered = [e for e in filtered if e.severity == severity]

        if client_id is not None:
            filtered = [e for e in filtered if e.client_id == client_id]

        if round_num is not None:
            filtered = [e for e in filtered if e.round_num == round_num]

        # Sort by timestamp (newest first)
        filtered = sorted(filtered, key=lambda e: e.timestamp, reverse=True)

        if limit is not None:
            filtered = filtered[:limit]

        return filtered

    def get_statistics(self) -> Dict[str, int]:
        """Get anomaly statistics."""
        return self.stats.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of anomalies."""
        total = len(self.events)

        if total == 0:
            return {
                "total_events": 0,
                "by_severity": {},
                "by_type": {},
                "unresolved": 0,
            }

        # Count by severity
        by_severity = {}
        for severity in AnomalySeverity:
            count = sum(1 for e in self.events if e.severity == severity)
            if count > 0:
                by_severity[severity.value] = count

        # Count by type
        by_type = {}
        for atype in AnomalyType:
            count = sum(1 for e in self.events if e.anomaly_type == atype)
            if count > 0:
                by_type[atype.value] = count

        # Count unresolved
        unresolved = sum(1 for e in self.events if not e.resolved)

        return {
            "total_events": total,
            "by_severity": by_severity,
            "by_type": by_type,
            "unresolved": unresolved,
        }

    def mark_resolved(self, event_indices: List[int]) -> None:
        """
        Mark events as resolved.

        Args:
            event_indices: Indices of events to resolve
        """
        for idx in event_indices:
            if 0 <= idx < len(self.events):
                self.events[idx].resolved = True

    def cleanup_old_events(self) -> None:
        """Remove events older than retention period."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=self.retention_days)

        original_count = len(self.events)
        self.events = [e for e in self.events if e.timestamp > cutoff]

        removed = original_count - len(self.events)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old anomaly events")

    def export_to_file(self, output_path: Path) -> None:
        """
        Export all events to a file.

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [event.to_dict() for event in self.events]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(data)} events to {output_path}")

    def import_from_file(self, input_path: Path) -> None:
        """
        Import events from a file.

        Args:
            input_path: Input file path
        """
        input_path = Path(input_path)

        if not input_path.exists():
            logger.warning(f"Import file not found: {input_path}")
            return

        with open(input_path, "r") as f:
            data = json.load(f)

        for event_dict in data:
            event = AnomalyEvent(
                timestamp=datetime.fromisoformat(event_dict["timestamp"]),
                anomaly_type=AnomalyType(event_dict["anomaly_type"]),
                severity=AnomalySeverity(event_dict["severity"]),
                client_id=event_dict.get("client_id"),
                round_num=event_dict.get("round_num"),
                message=event_dict.get("message", ""),
                details=event_dict.get("details", {}),
                confidence=event_dict.get("confidence", 0.0),
                resolved=event_dict.get("resolved", False),
            )
            self.events.append(event)
            self._update_stats(event)

        logger.info(f"Imported {len(data)} events from {input_path}")


class FLAnomalyLogger(AnomalyLogger):
    """
    Federated Learning specific anomaly logger.

    Adds FL context and specialized logging methods.
    """

    def log_poisoning_attack(
        self,
        client_ids: List[int],
        round_num: int,
        confidence: float,
        details: Optional[Dict] = None,
    ) -> None:
        """Log poisoning attack detection."""
        for client_id in client_ids:
            self.log(
                anomaly_type=AnomalyType.POISONING_ATTACK,
                severity=AnomalySeverity.HIGH if confidence > 0.7 else AnomalySeverity.MEDIUM,
                message=f"Potential poisoning attack detected",
                client_id=client_id,
                round_num=round_num,
                details=details or {},
                confidence=confidence,
            )

    def log_backdoor_attack(
        self,
        round_num: int,
        confidence: float,
        details: Optional[Dict] = None,
    ) -> None:
        """Log backdoor attack detection."""
        self.log(
            anomaly_type=AnomalyType.BACKDOOR_ATTACK,
            severity=AnomalySeverity.CRITICAL,
            message=f"Backdoor attack detected in global model",
            round_num=round_num,
            details=details or {},
            confidence=confidence,
        )

    def log_update_outlier(
        self,
        client_id: int,
        round_num: int,
        z_score: float,
        details: Optional[Dict] = None,
    ) -> None:
        """Log update outlier detection."""
        severity = (
            AnomalySeverity.HIGH if abs(z_score) > 5 else
            AnomalySeverity.MEDIUM if abs(z_score) > 3 else
            AnomalySeverity.LOW
        )

        self.log(
            anomaly_type=AnomalyType.UPDATE_OUTLIER,
            severity=severity,
            message=f"Update magnitude outlier detected (z-score: {z_score:.2f})",
            client_id=client_id,
            round_num=round_num,
            details=details or {},
            confidence=min(1.0, abs(z_score) / 3.0),
        )

    def log_performance_degradation(
        self,
        round_num: int,
        metric_name: str,
        degradation: float,
        details: Optional[Dict] = None,
    ) -> None:
        """Log performance degradation."""
        severity = (
            AnomalySeverity.HIGH if degradation > 0.2 else
            AnomalySeverity.MEDIUM if degradation > 0.1 else
            AnomalySeverity.LOW
        )

        self.log(
            anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
            severity=severity,
            message=f"Performance degradation: {metric_name} decreased by {degradation:.1%}",
            round_num=round_num,
            details=details or {},
            confidence=min(1.0, degradation * 5),
        )

    def log_communication_failure(
        self,
        client_id: int,
        round_num: int,
        error_message: str,
    ) -> None:
        """Log communication failure."""
        self.log(
            anomaly_type=AnomalyType.COMMUNICATION_FAILURE,
            severity=AnomalySeverity.MEDIUM,
            message=f"Communication failure: {error_message}",
            client_id=client_id,
            round_num=round_num,
            details={"error": error_message},
        )

    def get_client_risk_score(self, client_id: int) -> float:
        """
        Compute risk score for a client based on their anomaly history.

        Args:
            client_id: Client ID

        Returns:
            Risk score in [0, 1]
        """
        client_events = self.get_events(client_id=client_id)

        if not client_events:
            return 0.0

        # Weight events by severity and recency
        score = 0.0
        for event in client_events:
            # Severity weight
            severity_weight = {
                AnomalySeverity.LOW: 0.1,
                AnomalySeverity.MEDIUM: 0.3,
                AnomalySeverity.HIGH: 0.6,
                AnomalySeverity.CRITICAL: 1.0,
            }[event.severity]

            # Confidence
            score += severity_weight * event.confidence

        # Normalize to [0, 1]
        score = min(1.0, score / len(client_events))

        return score
