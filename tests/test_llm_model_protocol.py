"""TDD tests for LLMModel protocol and unified model abstractions.

These tests are written BEFORE the implementation (RED phase).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Protocol import tests
# ---------------------------------------------------------------------------


def test_llm_model_protocol_importable() -> None:
    from ecs_agent.providers.protocol import LLMModel  # noqa: F401


def test_llm_model_has_model_id_in_protocol() -> None:
    """LLMModel protocol must declare model_id as part of its interface."""
    import inspect
    from ecs_agent.providers.protocol import LLMModel

    members = {name for name, _ in inspect.getmembers(LLMModel)}
    assert "model_id" in members, "LLMModel protocol must expose 'model_id'"


def test_llm_provider_alias_still_importable() -> None:
    """LLMProvider must remain as a backward-compat alias."""
    from ecs_agent.providers.protocol import LLMProvider  # noqa: F401


# ---------------------------------------------------------------------------
# OpenAIModel (was OpenAIProvider)
# ---------------------------------------------------------------------------


def test_openai_model_importable() -> None:
    from ecs_agent.providers.openai_model import OpenAIModel  # noqa: F401


def test_openai_model_has_model_id_property() -> None:
    from ecs_agent.providers.config import ApiFormat, ProviderConfig
    from ecs_agent.providers.openai_model import OpenAIModel

    config = ProviderConfig(
        provider_id="test",
        base_url="https://api.example.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    m = OpenAIModel(config=config, model="gpt-4o")
    assert m.model_id == "gpt-4o"


def test_openai_model_satisfies_llm_model_protocol() -> None:
    from ecs_agent.providers.config import ApiFormat, ProviderConfig
    from ecs_agent.providers.openai_model import OpenAIModel
    from ecs_agent.providers.protocol import LLMModel

    config = ProviderConfig(
        provider_id="test",
        base_url="https://api.example.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    m = OpenAIModel(config=config, model="gpt-4o")
    assert isinstance(m, LLMModel)


def test_openai_provider_alias_still_works() -> None:
    """OpenAIProvider must remain importable as a backward-compat alias."""
    from ecs_agent.providers.openai_provider import OpenAIProvider  # noqa: F401


# ---------------------------------------------------------------------------
# ClaudeModel (was ClaudeProvider)
# ---------------------------------------------------------------------------


def test_claude_model_importable() -> None:
    from ecs_agent.providers.claude_model import ClaudeModel  # noqa: F401


def test_claude_model_has_model_id_property() -> None:
    from ecs_agent.providers.config import ApiFormat, ProviderConfig
    from ecs_agent.providers.claude_model import ClaudeModel

    config = ProviderConfig(
        provider_id="test",
        base_url="https://api.anthropic.com",
        api_key="test-key",
        api_format=ApiFormat.ANTHROPIC_MESSAGES,
    )
    m = ClaudeModel(config=config, model="claude-3-5-haiku-latest")
    assert m.model_id == "claude-3-5-haiku-latest"


def test_claude_model_satisfies_llm_model_protocol() -> None:
    from ecs_agent.providers.config import ApiFormat, ProviderConfig
    from ecs_agent.providers.claude_model import ClaudeModel
    from ecs_agent.providers.protocol import LLMModel

    config = ProviderConfig(
        provider_id="test",
        base_url="https://api.anthropic.com",
        api_key="test-key",
        api_format=ApiFormat.ANTHROPIC_MESSAGES,
    )
    m = ClaudeModel(config=config, model="claude-3-5-haiku-latest")
    assert isinstance(m, LLMModel)


def test_claude_provider_alias_still_works() -> None:
    from ecs_agent.providers.claude_provider import ClaudeProvider  # noqa: F401


# ---------------------------------------------------------------------------
# FakeModel (was FakeProvider)
# ---------------------------------------------------------------------------


def test_fake_model_importable() -> None:
    from ecs_agent.providers.fake_model import FakeModel  # noqa: F401


def test_fake_model_default_model_id() -> None:
    from ecs_agent.providers.fake_model import FakeModel

    m = FakeModel(responses=[])
    assert isinstance(m.model_id, str)
    assert len(m.model_id) > 0


def test_fake_model_custom_model_id() -> None:
    from ecs_agent.providers.fake_model import FakeModel

    m = FakeModel(responses=[], model_id="my-fake")
    assert m.model_id == "my-fake"


def test_fake_model_satisfies_llm_model_protocol() -> None:
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.providers.protocol import LLMModel

    m = FakeModel(responses=[])
    assert isinstance(m, LLMModel)


def test_fake_provider_alias_still_works() -> None:
    from ecs_agent.providers.fake_provider import FakeProvider  # noqa: F401


# ---------------------------------------------------------------------------
# RetryModel (was RetryProvider)
# ---------------------------------------------------------------------------


def test_retry_model_importable() -> None:
    from ecs_agent.providers.retry_model import RetryModel  # noqa: F401


def test_retry_model_delegates_model_id() -> None:
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.providers.retry_model import RetryModel

    inner = FakeModel(responses=[], model_id="inner-model")
    retry = RetryModel(model=inner)
    assert retry.model_id == "inner-model"


def test_retry_model_satisfies_llm_model_protocol() -> None:
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.providers.protocol import LLMModel
    from ecs_agent.providers.retry_model import RetryModel

    inner = FakeModel(responses=[])
    retry = RetryModel(model=inner)
    assert isinstance(retry, LLMModel)


def test_retry_provider_alias_still_works() -> None:
    from ecs_agent.providers.retry_provider import RetryProvider  # noqa: F401


# ---------------------------------------------------------------------------
# LLMComponent: model field is now LLMModel, not str
# ---------------------------------------------------------------------------


def test_llm_component_accepts_llm_model_object() -> None:
    from ecs_agent.components.definitions import LLMComponent
    from ecs_agent.providers.fake_model import FakeModel

    m = FakeModel(responses=[])
    comp = LLMComponent(model=m)
    assert comp.model is m


def test_llm_component_model_id_via_model_dot_model_id() -> None:
    from ecs_agent.components.definitions import LLMComponent
    from ecs_agent.providers.fake_model import FakeModel

    m = FakeModel(responses=[], model_id="test-id")
    comp = LLMComponent(model=m)
    assert comp.model.model_id == "test-id"


def test_llm_component_pending_model_is_llm_model_or_none() -> None:
    from ecs_agent.components.definitions import LLMComponent
    from ecs_agent.providers.fake_model import FakeModel

    m1 = FakeModel(responses=[])
    m2 = FakeModel(responses=[], model_id="pending")
    comp = LLMComponent(model=m1, pending_model=m2)
    assert comp.pending_model is m2
    assert comp.pending_model.model_id == "pending"


def test_llm_component_has_no_provider_field() -> None:
    """After refactoring, LLMComponent must NOT have a 'provider' field."""
    from ecs_agent.components.definitions import LLMComponent
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(LLMComponent)}
    assert "provider" not in field_names, (
        "LLMComponent still has 'provider' field — should be removed in refactoring"
    )
