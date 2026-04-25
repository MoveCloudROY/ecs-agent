"""Backward-compat alias — use claude_model instead."""
from ecs_agent.providers.claude_model import ClaudeModel as ClaudeProvider  # noqa: F401

__all__ = ["ClaudeProvider"]
