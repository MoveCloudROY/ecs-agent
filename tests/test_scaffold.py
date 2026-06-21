"""Tests for project scaffolding and basic imports."""

import ecs_agent
import tomllib
from pathlib import Path


def test_version_matches_pyproject() -> None:
    """ecs_agent.__version__ should match the version in pyproject.toml."""
    project_root = Path(__file__).parent.parent
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())
    expected_version = pyproject["project"]["version"]
    assert ecs_agent.__version__ == expected_version
