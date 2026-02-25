"""Providers module for LLM integrations."""

from typing import Any

from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.providers.openai_provider import OpenAIProvider
from ecs_agent.providers.claude_provider import ClaudeProvider

try:
    from ecs_agent.providers.litellm_provider import LiteLLMProvider
except ImportError:
    LiteLLMProvider = None  # type: ignore[assignment, misc]
__all__ = [
    "LLMProvider",
    "FakeProvider",
    "OpenAIProvider",
    "ClaudeProvider",
    "LiteLLMProvider",
]
