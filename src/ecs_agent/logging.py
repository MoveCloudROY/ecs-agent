"""Structured logging configuration using structlog."""

import inspect
import logging
import os
import sys
from typing import Any

import structlog
from structlog.contextvars import merge_contextvars
from structlog.processors import (
    JSONRenderer,
    TimeStamper,
    add_log_level,
    format_exc_info,
)
from structlog.dev import ConsoleRenderer

_log_level = "INFO"
_module_levels: dict[str, int] = {}
_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_METHOD_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "exception": logging.ERROR,
}


def _add_caller_info(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    del logger
    del method_name
    frame = inspect.currentframe()
    if frame is None:
        return event_dict

    caller_frame = frame.f_back
    while caller_frame is not None:
        module_name = str(caller_frame.f_globals.get("__name__", ""))
        filename = caller_frame.f_code.co_filename
        if (
            module_name.startswith("structlog")
            or module_name.startswith("logging")
            or filename == __file__
        ):
            caller_frame = caller_frame.f_back
            continue

        event_dict["caller_file"] = os.path.basename(filename)
        event_dict["caller_line"] = caller_frame.f_lineno
        event_dict["caller_function"] = caller_frame.f_code.co_name
        break

    return event_dict


def _filter_by_module_level(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    if not _module_levels:
        return event_dict

    logger_name = str(event_dict.get("logger") or getattr(logger, "name", ""))
    if not logger_name:
        return event_dict

    matched_level: int | None = None
    matched_prefix_len = -1
    for module_name, module_level in _module_levels.items():
        if logger_name == module_name or logger_name.startswith(f"{module_name}."):
            if len(module_name) > matched_prefix_len:
                matched_level = module_level
                matched_prefix_len = len(module_name)

    if matched_level is None:
        return event_dict

    method_level = _METHOD_LEVEL_MAP.get(method_name, logging.INFO)
    if method_level >= matched_level:
        return event_dict
    raise structlog.DropEvent()


def _ensure_exc_info(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    del logger
    if method_name == "exception" and "exc_info" not in event_dict:
        event_dict["exc_info"] = True
    return event_dict


def _filter_by_level(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Filter events by configured log level."""
    configured_level = _LEVEL_MAP.get(_log_level, logging.INFO)
    method_level = _METHOD_LEVEL_MAP.get(method_name, logging.INFO)
    if method_level >= configured_level:
        return event_dict
    raise structlog.DropEvent()


def configure_logging(
    json_output: bool = False,
    level: str = "INFO",
    module_levels: dict[str, str] | None = None,
    colors: bool = True,
) -> None:
    """Configure structlog with processors for structured logging.

    Args:
        json_output: If True, output JSON format (production). If False, use console format (development).
        level: Logging level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        module_levels: Optional module-specific level map. Keys are logger/module prefixes and values are levels.
        colors: Whether to use colored output in console mode.
    """
    global _log_level
    global _module_levels
    _log_level = level
    _module_levels = {
        module_name: _LEVEL_MAP.get(module_level.upper(), logging.INFO)
        for module_name, module_level in (module_levels or {}).items()
    }

    shared_processors: list[Any] = [
        merge_contextvars,
        _add_caller_info,
        _filter_by_module_level,
        _filter_by_level,
        add_log_level,
        TimeStamper(fmt="iso"),
        _ensure_exc_info,
        format_exc_info,
    ]
    foreign_processors: list[Any] = [
        merge_contextvars,
        _add_caller_info,
        add_log_level,
        TimeStamper(fmt="iso"),
        _ensure_exc_info,
        format_exc_info,
    ]

    renderer = JSONRenderer() if json_output else ConsoleRenderer(colors=colors)
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=foreign_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(_LEVEL_MAP.get(_log_level, logging.INFO))

    processors: list[Any] = [*shared_processors, renderer]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Get a bound logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of calling module).

    Returns:
        A structlog bound logger instance.
    """
    return structlog.get_logger(name).bind(logger=name)
