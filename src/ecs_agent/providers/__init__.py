"""Providers module for LLM integrations."""

from ecs_agent.providers.config import ApiFormat, ProviderConfig, ProviderEntry
from ecs_agent.providers.model_factory import create_model
from ecs_agent.providers.model_id import ModelId, format_model_id, parse_model_id
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.providers.openai_files import OpenAIFilesService
from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.providers.registry import ProviderRegistry, get_model
from ecs_agent.providers.retry_model import RetryModel

try:
    from ecs_agent.providers.litellm_model import LiteLLMModel
except ImportError:
    LiteLLMModel = None  # type: ignore[assignment, misc]

__all__ = [
    "LLMModel",
    "OpenAIModel",
    "ClaudeModel",
    "FakeModel",
    "RetryModel",
    "LiteLLMModel",
    "create_model",
    "get_model",
    "ApiFormat",
    "ProviderConfig",
    "ProviderEntry",
    "ProviderRegistry",
    "ModelId",
    "parse_model_id",
    "format_model_id",
    "OpenAIFilesService",
]
