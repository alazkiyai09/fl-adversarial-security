"""Alerting system for security events.

Supports multiple channels:
- Log-based alerts
- Email alerts
- Webhook alerts (Slack, Teams, etc.)
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
import json
from pathlib import Path

import requests
from loguru import logger

from .anomaly_logger import AnomalyEvent, AnomalySeverity


class AlertChannel(Enum):
    """Alert channel types."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    CUSTOM = "custom"


@dataclass
class AlertConfig:
    """Configuration for alert channel."""
    channel: AlertChannel
    enabled: bool = True
    min_severity: AnomalySeverity = AnomalySeverity.MEDIUM
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}


class AlertManager:
    """
    Manages alert sending for security events.

    Supports:
    - Multiple alert channels
    - Severity filtering
    - Rate limiting
    - Custom alert formatting
    """

    def __init__(
        self,
        channels: Optional[List[AlertConfig]] = None,
        rate_limit_minutes: int = 5,
    ):
        """
        Initialize alert manager.

        Args:
            channels: List of alert channel configurations
            rate_limit_minutes: Minimum minutes between alerts for same event type
        """
        self.channels = channels or []
        self.rate_limit_minutes = rate_limit_minutes

        # Rate limiting tracking
        self.last_alert_time: Dict[str, float] = {}

        # Custom formatters
        self.formatters: Dict[AlertChannel, Callable] = {}

    def add_channel(self, config: AlertConfig) -> None:
        """
        Add an alert channel.

        Args:
            config: Channel configuration
        """
        self.channels.append(config)
        logger.info(f"Added alert channel: {config.channel.value}")

    def set_formatter(
        self,
        channel: AlertChannel,
        formatter: Callable[[AnomalyEvent], str],
    ) -> None:
        """
        Set custom formatter for a channel.

        Args:
            channel: Channel type
            formatter: Formatter function
        """
        self.formatters[channel] = formatter

    def send_alert(self, event: AnomalyEvent) -> bool:
        """
        Send alert for an event.

        Args:
            event: Anomaly event

        Returns:
            True if alert was sent to at least one channel
        """
        # Check if should send based on severity
        if not self._should_alert(event):
            return False

        # Check rate limiting
        if self._is_rate_limited(event):
            logger.debug(f"Alert rate-limited for {event.anomaly_type.value}")
            return False

        # Send to all enabled channels
        sent = False
        for channel_config in self.channels:
            if not channel_config.enabled:
                continue

            # Check severity threshold
            if self._severity_below_threshold(event, channel_config):
                continue

            try:
                if self._send_to_channel(event, channel_config):
                    sent = True
            except Exception as e:
                logger.error(f"Failed to send alert to {channel_config.channel.value}: {e}")

        # Update rate limit tracking
        if sent:
            self.last_alert_time[event.anomaly_type.value] = event.timestamp.timestamp()

        return sent

    def _should_alert(self, event: AnomalyEvent) -> bool:
        """Check if event should trigger an alert."""
        # Don't alert for resolved events
        if event.resolved:
            return False

        # Don't alert for low severity unless configured
        return event.severity != AnomalySeverity.LOW

    def _is_rate_limited(self, event: AnomalyEvent) -> bool:
        """Check if alert is rate limited."""
        key = event.anomaly_type.value

        if key not in self.last_alert_time:
            return False

        elapsed = event.timestamp.timestamp() - self.last_alert_time[key]
        rate_limit_seconds = self.rate_limit_minutes * 60

        return elapsed < rate_limit_seconds

    def _severity_below_threshold(
        self,
        event: AnomalyEvent,
        channel_config: AlertConfig,
    ) -> bool:
        """Check if event severity is below channel threshold."""
        severity_order = {
            AnomalySeverity.LOW: 0,
            AnomalySeverity.MEDIUM: 1,
            AnomalySeverity.HIGH: 2,
            AnomalySeverity.CRITICAL: 3,
        }

        return severity_order[event.severity] < severity_order[channel_config.min_severity]

    def _send_to_channel(
        self,
        event: AnomalyEvent,
        channel_config: AlertConfig,
    ) -> bool:
        """Send alert to specific channel."""
        if channel_config.channel == AlertChannel.LOG:
            return self._send_log_alert(event, channel_config)
        elif channel_config.channel == AlertChannel.EMAIL:
            return self._send_email_alert(event, channel_config)
        elif channel_config.channel == AlertChannel.WEBHOOK:
            return self._send_webhook_alert(event, channel_config)
        elif channel_config.channel == AlertChannel.SLACK:
            return self._send_slack_alert(event, channel_config)
        elif channel_config.channel == AlertChannel.TEAMS:
            return self._send_teams_alert(event, channel_config)
        elif channel_config.channel == AlertChannel.CUSTOM:
            return self._send_custom_alert(event, channel_config)
        else:
            logger.warning(f"Unknown alert channel: {channel_config.channel}")
            return False

    def _send_log_alert(self, event: AnomalyEvent, config: AlertConfig) -> bool:
        """Send log alert (already logged by AnomalyLogger)."""
        # This is a no-op since AnomalyLogger already handles console logging
        return True

    def _send_email_alert(self, event: AnomalyEvent, config: AlertConfig) -> bool:
        """Send email alert."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Get configuration
            smtp_server = config.config.get("smtp_server")
            smtp_port = config.config.get("smtp_port", 587)
            username = config.config.get("username")
            password = config.config.get("password")
            from_addr = config.config.get("from_addr")
            to_addrs = config.config.get("to_addrs", [])

            if not all([smtp_server, username, password, from_addr, to_addrs]):
                logger.warning("Incomplete email configuration")
                return False

            # Format message
            subject = f"[{event.severity.value.upper()}] {event.anomaly_type.value} detected"
            body = self._format_alert_message(event)

            # Send email
            msg = MIMEMultipart()
            msg["From"] = from_addr
            msg["To"] = ", ".join(to_addrs)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)

            logger.info(f"Email alert sent to {to_addrs}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _send_webhook_alert(self, event: AnomalyEvent, config: AlertConfig) -> bool:
        """Send webhook alert."""
        url = config.config.get("url")
        if not url:
            logger.warning("Webhook URL not configured")
            return False

        # Prepare payload
        payload = {
            "timestamp": event.timestamp.isoformat(),
            "type": event.anomaly_type.value,
            "severity": event.severity.value,
            "message": event.message,
            "client_id": event.client_id,
            "round_num": event.round_num,
            "confidence": event.confidence,
            "details": event.details,
        }

        # Send webhook
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()

            logger.info(f"Webhook alert sent to {url}")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    def _send_slack_alert(self, event: AnomalyEvent, config: AlertConfig) -> bool:
        """Send Slack alert via webhook."""
        webhook_url = config.config.get("webhook_url")
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        # Color based on severity
        color_map = {
            AnomalySeverity.LOW: "#36a64f",  # green
            AnomalySeverity.MEDIUM: "#ff9900",  # orange
            AnomalySeverity.HIGH: "#ff0000",  # red
            AnomalySeverity.CRITICAL: "#990000",  # dark red
        }

        # Format message
        client_str = f"Client {event.client_id}" if event.client_id is not None else "Server"
        round_str = f"Round {event.round_num}" if event.round_num is not None else ""

        attachment = {
            "color": color_map.get(event.severity, "#808080"),
            "title": f"{event.severity.value.upper()}: {event.anomaly_type.value}",
            "text": event.message,
            "fields": [
                {"title": "Client", "value": client_str, "short": True},
                {"title": "Round", "value": round_str, "short": True},
                {"title": "Confidence", "value": f"{event.confidence:.2f}", "short": True},
            ],
            "footer": "FL Security Alert",
            "ts": int(event.timestamp.timestamp()),
        }

        # Send to Slack
        try:
            response = requests.post(
                webhook_url,
                json={"attachments": [attachment]},
                timeout=10,
            )
            response.raise_for_status()

            logger.info("Slack alert sent")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _send_teams_alert(self, event: AnomalyEvent, config: AlertConfig) -> bool:
        """Send Microsoft Teams alert via webhook."""
        webhook_url = config.config.get("webhook_url")
        if not webhook_url:
            logger.warning("Teams webhook URL not configured")
            return False

        # Color based on severity
        color_map = {
            AnomalySeverity.LOW: "00FF00",
            AnomalySeverity.MEDIUM: "FF9900",
            AnomalySeverity.HIGH: "FF0000",
            AnomalySeverity.CRITICAL: "990000",
        }

        # Format message
        client_str = f"Client {event.client_id}" if event.client_id is not None else "Server"
        round_str = f"Round {event.round_num}" if event.round_num is not None else "N/A"

        card = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": f"{event.anomaly_type.value} detected",
            "themeColor": color_map.get(event.severity, "808080"),
            "title": f"{event.severity.value.upper()}: {event.anomaly_type.value}",
            "text": event.message,
            "sections": [{
                "facts": [
                    {"name": "Client", "value": client_str},
                    {"name": "Round", "value": round_str},
                    {"name": "Confidence", "value": f"{event.confidence:.2f}"},
                    {"name": "Time", "value": event.timestamp.strftime("%Y-%m-%d %H:%M:%S")},
                ],
            }],
        }

        # Send to Teams
        try:
            response = requests.post(
                webhook_url,
                json=card,
                timeout=10,
            )
            response.raise_for_status()

            logger.info("Teams alert sent")
            return True

        except Exception as e:
            logger.error(f"Failed to send Teams alert: {e}")
            return False

    def _send_custom_alert(self, event: AnomalyEvent, config: AlertConfig) -> bool:
        """Send custom alert using user-provided function."""
        handler = config.config.get("handler")
        if not handler or not callable(handler):
            logger.warning("Custom alert handler not provided or not callable")
            return False

        try:
            result = handler(event)
            return bool(result)
        except Exception as e:
            logger.error(f"Custom alert handler failed: {e}")
            return False

    def _format_alert_message(self, event: AnomalyEvent) -> str:
        """Format alert message."""
        lines = [
            f"Security Alert: {event.anomaly_type.value}",
            f"Severity: {event.severity.value}",
            f"Message: {event.message}",
            "",
        ]

        if event.client_id is not None:
            lines.append(f"Client ID: {event.client_id}")

        if event.round_num is not None:
            lines.append(f"Round: {event.round_num}")

        lines.append(f"Confidence: {event.confidence:.2f}")
        lines.append(f"Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        if event.details:
            lines.append("\nDetails:")
            lines.append(json.dumps(event.details, indent=2))

        return "\n".join(lines)


def create_alert_manager(config: Dict) -> AlertManager:
    """
    Create alert manager from configuration.

    Args:
        config: Alert configuration

    Returns:
        AlertManager instance
    """
    manager = AlertManager(
        rate_limit_minutes=config.get("rate_limit_minutes", 5),
    )

    # Add log channel (always enabled)
    manager.add_channel(AlertConfig(
        channel=AlertChannel.LOG,
        enabled=True,
        min_severity=AnomalySeverity.MEDIUM,
    ))

    # Add email if configured
    if config.get("email", {}).get("enabled"):
        manager.add_channel(AlertConfig(
            channel=AlertChannel.EMAIL,
            enabled=True,
            min_severity=AnomalySeverity.HIGH,
            config=config["email"],
        ))

    # Add webhook if configured
    if config.get("webhook", {}).get("enabled"):
        manager.add_channel(AlertConfig(
            channel=AlertChannel.WEBHOOK,
            enabled=True,
            min_severity=AnomalySeverity.HIGH,
            config=config["webhook"],
        ))

    # Add Slack if configured
    if config.get("slack", {}).get("enabled"):
        manager.add_channel(AlertConfig(
            channel=AlertChannel.SLACK,
            enabled=True,
            min_severity=AnomalySeverity.MEDIUM,
            config=config["slack"],
        ))

    # Add Teams if configured
    if config.get("teams", {}).get("enabled"):
        manager.add_channel(AlertConfig(
            channel=AlertChannel.TEAMS,
            enabled=True,
            min_severity=AnomalySeverity.MEDIUM,
            config=config["teams"],
        ))

    logger.info(f"Alert manager created with {len(manager.channels)} channels")

    return manager
