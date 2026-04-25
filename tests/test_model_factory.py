"""TDD tests for create_model() factory and updated get_model() registry function."""

from __future__ import annotations

import pytest

from ecs_agent.providers.config import ApiFormat, ProviderConfig


def _make_config(api_format: ApiFormat) -> ProviderConfig:
    return ProviderConfig(
        provider_id="test",
        base_url="https://api.example.com/v1",
        api_key="test-key",
        api_format=api_format,
    )


# ---------------------------------------------------------------------------
# create_model() factory
# ---------------------------------------------------------------------------


def test_create_model_importable() -> None:
    from ecs_agent.providers.model_factory import create_model  # noqa: F401


def test_create_model_openai_chat_completions_returns_openai_model() -> None:
    from ecs_agent.providers.model_factory import create_model
    from ecs_agent.providers.openai_model import OpenAIModel

    m = create_model(_make_config(ApiFormat.OPENAI_CHAT_COMPLETIONS), "gpt-4o")
    assert isinstance(m, OpenAIModel)


def test_create_model_openai_responses_returns_openai_model() -> None:
    from ecs_agent.providers.model_factory import create_model
    from ecs_agent.providers.openai_model import OpenAIModel

    m = create_model(_make_config(ApiFormat.OPENAI_RESPONSES), "gpt-4o")
    assert isinstance(m, OpenAIModel)


def test_create_model_anthropic_messages_returns_claude_model() -> None:
    from ecs_agent.providers.claude_model import ClaudeModel
    from ecs_agent.providers.model_factory import create_model

    m = create_model(_make_config(ApiFormat.ANTHROPIC_MESSAGES), "claude-3-5-haiku")
    assert isinstance(m, ClaudeModel)


def test_create_model_unsupported_format_raises() -> None:
    from ecs_agent.providers.model_factory import create_model

    with pytest.raises(ValueError, match="not supported"):
        create_model(_make_config(ApiFormat.OPENAI_EMBEDDINGS), "text-embedding-3")


def test_create_model_model_id_matches_input() -> None:
    from ecs_agent.providers.model_factory import create_model

    m = create_model(_make_config(ApiFormat.OPENAI_CHAT_COMPLETIONS), "my-model-42")
    assert m.model_id == "my-model-42"


def test_create_model_returns_llm_model_protocol() -> None:
    from ecs_agent.providers.model_factory import create_model
    from ecs_agent.providers.protocol import LLMModel

    m = create_model(_make_config(ApiFormat.OPENAI_CHAT_COMPLETIONS), "gpt-4o")
    assert isinstance(m, LLMModel)


# ---------------------------------------------------------------------------
# get_model() registry function
# ---------------------------------------------------------------------------


def test_get_model_importable() -> None:
    from ecs_agent.providers.registry import get_model  # noqa: F401


def test_get_model_returns_llm_model() -> None:
    from ecs_agent.providers.protocol import LLMModel
    from ecs_agent.providers.registry import ProviderRegistry, get_model

    registry = ProviderRegistry.from_dict(
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_format": "openai_chat_completions",
                "api_key": "test-key",
            }
        }
    )
    m = get_model("openai/gpt-4o", registry=registry)
    assert isinstance(m, LLMModel)


def test_get_model_openai_returns_openai_model() -> None:
    from ecs_agent.providers.openai_model import OpenAIModel
    from ecs_agent.providers.registry import ProviderRegistry, get_model

    registry = ProviderRegistry.from_dict(
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_format": "openai_chat_completions",
                "api_key": "test-key",
            }
        }
    )
    m = get_model("openai/gpt-4o", registry=registry)
    assert isinstance(m, OpenAIModel)


def test_get_model_anthropic_returns_claude_model() -> None:
    from ecs_agent.providers.claude_model import ClaudeModel
    from ecs_agent.providers.registry import ProviderRegistry, get_model

    registry = ProviderRegistry.from_dict(
        {
            "anthropic": {
                "base_url": "https://api.anthropic.com",
                "api_format": "anthropic_messages",
                "api_key": "test-key",
            }
        }
    )
    m = get_model("anthropic/claude-3-5-haiku", registry=registry)
    assert isinstance(m, ClaudeModel)


def test_get_model_model_id_matches_requested_model() -> None:
    from ecs_agent.providers.registry import ProviderRegistry, get_model

    registry = ProviderRegistry.from_dict(
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_format": "openai_chat_completions",
                "api_key": "test-key",
            }
        }
    )
    m = get_model("openai/gpt-4-turbo", registry=registry)
    assert m.model_id == "gpt-4-turbo"

