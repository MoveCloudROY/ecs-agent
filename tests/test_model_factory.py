"""Regression tests for public model construction APIs."""

from __future__ import annotations

import pytest

import importlib

from ecs_agent.providers.config import ApiFormat


def test_model_is_exported_from_root_package() -> None:
    root_module = importlib.import_module("ecs_agent")

    assert hasattr(root_module, "Model")


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


def test_get_model_openai_responses_retains_responses_api_format() -> None:
    from ecs_agent.providers.registry import ProviderRegistry, get_model

    registry = ProviderRegistry.from_dict(
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_format": "openai_responses",
                "api_key": "test-key",
            }
        }
    )
    m = get_model("openai/gpt-4o", registry=registry)

    assert m._provider_config.api_format is ApiFormat.OPENAI_RESPONSES
