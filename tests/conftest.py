"""Pytest configuration and shared fixtures."""

import pytest

from ecs_agent.logging import configure_logging


@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """Configure logging for all tests at session start."""
    # Default to console output for test visibility
    # Individual tests can reconfigure as needed
    configure_logging(json_output=False, level="DEBUG")
