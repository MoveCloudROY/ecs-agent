"""Pytest configuration and shared fixtures."""

import structlog
import pytest


def pytest_configure(config):
    """Configure logging before any test collection."""
    # Import and call configure_logging from ecs_agent
    from ecs_agent.logging import configure_logging
    
    # Configure with JSON output for tests
    configure_logging(json_output=True, level="INFO")
    
    # Reconfigure structlog to DISABLE caching
    # This allows loggers created at module import time to be reconfigured
    current_config = structlog.get_config()
    structlog.configure(
        processors=current_config["processors"],
        context_class=current_config["context_class"],
        logger_factory=current_config["logger_factory"],
        cache_logger_on_first_use=False,  # Disable caching for tests
    )
