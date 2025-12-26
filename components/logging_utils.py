"""
Structured logging with session/request ID tracking and cost logging.
Includes secret redaction for security.
"""

import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler
import uuid

from components.config import Config
from components.security import create_safe_log_entry, redact_dict


class SessionIDFilter(logging.Filter):
    """Add session_id and request_id to all log records."""

    def __init__(self):
        super().__init__()
        self.session_id = str(uuid.uuid4())[:8]
        self.request_id: Optional[str] = None

    def filter(self, record):
        """Add session/request IDs to log record."""
        record.session_id = self.session_id
        record.request_id = self.request_id or "N/A"
        return True


# Global session filter
_session_filter = SessionIDFilter()


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging with secret redaction."""

    def format(self, record):
        """Format log record as JSON with secrets redacted."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            # Redact secrets
            "message": create_safe_log_entry(record.getMessage()),
            "session_id": getattr(record, "session_id", "N/A"),
            "request_id": getattr(record, "request_id", "N/A"),
        }

        # Add extra fields
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "tokens_in"):
            log_data["tokens_in"] = record.tokens_in
        if hasattr(record, "tokens_out"):
            log_data["tokens_out"] = record.tokens_out
        if hasattr(record, "cost_usd"):
            log_data["cost_usd"] = record.cost_usd
        if hasattr(record, "agent"):
            log_data["agent"] = record.agent
        if hasattr(record, "model"):
            log_data["model"] = record.model

        # Add exception info if present (redacted)
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            log_data["exception"] = create_safe_log_entry(exc_text)

        return json.dumps(log_data)


class PlainFormatter(logging.Formatter):
    """Format logs as plain text (for console)."""

    def format(self, record):
        """Format log record as plain text."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        session_id = getattr(record, "session_id", "N/A")[:8]
        request_id = getattr(record, "request_id", "N/A")[:8]

        msg = f"[{timestamp}] {record.levelname:8} [session:{session_id}] {record.getMessage()}"

        # Add metrics if present
        metrics = []
        if hasattr(record, "latency_ms"):
            metrics.append(f"latency:{record.latency_ms:.0f}ms")
        if hasattr(record, "tokens_in"):
            metrics.append(f"tokens_in:{record.tokens_in}")
        if hasattr(record, "cost_usd"):
            metrics.append(f"cost:${record.cost_usd:.4f}")

        if metrics:
            msg += f" [{', '.join(metrics)}]"

        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return msg


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    use_json: bool = False,
    session_filter: Optional[SessionIDFilter] = None,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        log_file: File to write logs to (optional)
        use_json: Format logs as JSON (default: plain text)
        session_filter: Session filter to add context

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Don't reconfigure if already configured
    if logger.handlers:
        return logger

    # Set log level
    log_level = getattr(logging, Config.LOG_LEVEL)
    logger.setLevel(log_level)

    # Use global session filter if not provided
    if session_filter is None:
        session_filter = _session_filter

    logger.addFilter(session_filter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = JSONFormatter() if use_json else PlainFormatter()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log file specified or using default)
    log_file = log_file or Config.LOG_FILE
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
            )
            file_handler.setLevel(log_level)
            formatter = JSONFormatter() if use_json else PlainFormatter()
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(
                f"Could not create file handler for {log_file}: {e}")

    return logger


def log_api_call(
    logger: logging.Logger,
    model: str,
    agent: str,
    latency_ms: float,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    cost_usd: float = 0.0,
):
    """
    Log an API call with metrics.

    Args:
        logger: Logger instance
        model: Model name (e.g., "openai/gpt-4o")
        agent: Agent name (e.g., "visual")
        latency_ms: API call latency in milliseconds
        tokens_in: Input tokens (optional)
        tokens_out: Output tokens (optional)
        cost_usd: Estimated cost in USD
    """
    total_tokens = (tokens_in or 0) + (tokens_out or 0)
    msg = f"API call: {model} ({agent}) - {latency_ms:.0f}ms"

    if total_tokens > 0:
        msg += f" - {total_tokens} tokens"
    if cost_usd > 0:
        msg += f" - ${cost_usd:.6f}"

    extra = {
        "model": model,
        "agent": agent,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
    }

    if tokens_in is not None:
        extra["tokens_in"] = tokens_in
    if tokens_out is not None:
        extra["tokens_out"] = tokens_out

    logger.info(msg, extra=extra)


def set_request_id(request_id: str):
    """Set current request ID for logging context."""
    _session_filter.request_id = request_id


def get_session_id() -> str:
    """Get current session ID."""
    return _session_filter.session_id


# Create default logger for this module
logger = get_logger(__name__)
