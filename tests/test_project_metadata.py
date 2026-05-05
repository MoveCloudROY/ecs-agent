"""Project metadata tests."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_langfuse_extra_includes_socks_proxy_support() -> None:
    """ecs-agent[langfuse] installs httpx SOCKS support for proxied Langfuse use."""
    project_root = Path(__file__).parent.parent
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())

    langfuse_extra = pyproject["project"]["optional-dependencies"]["langfuse"]

    assert any(dependency.startswith("httpx[socks]") for dependency in langfuse_extra)
