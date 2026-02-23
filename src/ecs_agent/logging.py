"""Structured logging configuration using structlog."""

import logging
import structlog
from structlog.contextvars import merge_contextvars
from structlog.processors import add_log_level, TimeStamper, JSONRenderer
from structlog.dev import ConsoleRenderer

_log_level = "INFO"


def _filter_by_level(logger, method_name, event_dict):
    """Filter events by configured log level."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    method_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    configured_level = level_map.get(_log_level, logging.INFO)
    method_level = method_level_map.get(method_name, logging.INFO)
    if method_level >= configured_level:
        return event_dict
    raise structlog.DropEvent()


def configure_logging(json_output: bool = False, level: str = "INFO"):
    """Configure structlog with processors for structured logging.

    Args:
        json_output: If True, output JSON format (production). If False, use console format (development).
        level: Logging level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    global _log_level
    _log_level = level

    processors = [
        merge_contextvars,
        _filter_by_level,
        add_log_level,
        TimeStamper(fmt="iso"),
        JSONRenderer() if json_output else ConsoleRenderer(),
    ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Get a bound logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of calling module).

    Returns:
        A structlog bound logger instance.
    """
    return structlog.get_logger(name)
