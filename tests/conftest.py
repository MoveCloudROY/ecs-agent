"""Pytest configuration and shared fixtures."""

# Configure logging IMMEDIATELY before any ECS imports happen
import structlog
import logging
import sys
from structlog.processors import (
    JSONRenderer,
    TimeStamper,
    add_log_level,
    format_exc_info,
)
from structlog.contextvars import merge_contextvars

# Minimal processors for test environment
def _ensure_exc_info_minimal(logger, method_name, event_dict):
    if method_name == "exception" and "exc_info" not in event_dict:
        event_dict["exc_info"] = True
    return event_dict

# Configure JSON logging BEFORE any ecs_agent imports
shared_processors = [
    merge_contextvars,
    add_log_level,
    TimeStamper(fmt="iso"),
    _ensure_exc_info_minimal,
    format_exc_info,
]

renderer = JSONRenderer()
processors = [*shared_processors, renderer]

structlog.configure(
    processors=processors,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

# Configure stdlib logging bridge
handler = logging.StreamHandler(stream=sys.stdout)
formatter = structlog.stdlib.ProcessorFormatter(
    foreign_pre_chain=shared_processors,
    processors=[
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        renderer,
    ],
)
handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

# Now safe to import pytest and ecs_agent
import pytest
