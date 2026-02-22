"""Tests for project scaffolding and basic imports."""

import ecs_agent


def test_import_ecs_agent() -> None:
    """Test that ecs_agent can be imported successfully."""
    assert ecs_agent is not None


def test_version_exists() -> None:
    """Test that ecs_agent.__version__ exists and has correct value."""
    assert hasattr(ecs_agent, "__version__")
    assert ecs_agent.__version__ == "0.1.0"
