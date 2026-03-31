"""Providers module for LLM integrations."""

from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.model_id import ModelId, format_model_id, parse_model_id
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.providers.openai_provider import OpenAIProvider
from ecs_agent.providers.openai_files import OpenAIFilesService
from ecs_agent.providers.claude_provider import ClaudeProvider

try:
    from ecs_agent.providers.litellm_provider import LiteLLMProvider
except ImportError:
    LiteLLMProvider = None  # type: ignore[assignment, misc]
__all__ = [
    "LLMProvider",
    "ApiFormat",
    "ProviderConfig",
    "ModelId",
    "parse_model_id",
    "format_model_id",
    "FakeProvider",
    "OpenAIProvider",
    "OpenAIFilesService",
    "ClaudeProvider",
    "LiteLLMProvider",
]
