"""Strict placeholder renderer for bounded grammar substitution.

Provides deterministic placeholder expansion using Python's string.Template
with no eval, no code execution, and explicit error handling.
"""

from __future__ import annotations

from ecs_agent.placeholder.renderer import StrictPlaceholderRenderer

__all__ = ["StrictPlaceholderRenderer"]
