"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to Loguru."""
        # Get corresponding Loguru level if it exists
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """
    Configure logging for the application.

    Args:
        log_file: Path to log file (optional, console only if None)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: Log rotation settings (e.g., "10 MB", "1 day")
        retention: Log retention period
        format_string: Log message format
    """
    # Remove default handler
    loguru_logger.remove()

    # Add console handler with color
    loguru_logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        loguru_logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            enqueue=True,  # Async logging
        )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def get_logger(name: str):
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return loguru_logger.bind(name=name)


class ContextLogger:
    """Context-aware logger for federated learning operations."""

    def __init__(self, name: str):
        """
        Initialize context logger.

        Args:
            name: Logger name
        """
        self.logger = loguru_logger.bind(name=name)
        self.context = {}

    def with_context(self, **kwargs) -> "ContextLogger":
        """
        Add context to logger.

        Args:
            **kwargs: Context key-value pairs

        Returns:
            Self with updated context
        """
        self.context.update(kwargs)
        return self

    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{context_str} | {message}"
        return message

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message))

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message))

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message))

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message))

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(message))


def get_fl_logger(
    node_type: str, node_id: Optional[int] = None, round_num: Optional[int] = None
) -> ContextLogger:
    """
    Get a context-aware logger for federated learning.

    Args:
        node_type: Type of node (server, client)
        node_id: Node ID (for clients)
        round_num: Current round number

    Returns:
        ContextLogger with FL context
    """
    logger = ContextLogger("fl")
    logger.with_context(node_type=node_type)

    if node_id is not None:
        logger.with_context(node_id=node_id)

    if round_num is not None:
        logger.with_context(round=round_num)

    return logger
