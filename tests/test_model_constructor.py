"""TDD tests for the unified Model(...) constructor interface."""

from __future__ import annotations

import pytest

from ecs_agent.providers.config import ApiFormat


# ---------------------------------------------------------------------------
# 1. Importability
# ---------------------------------------------------------------------------


def test_model_constructor_importable() -> None:
    from ecs_agent.providers.model_constructor import Model  # noqa: F401


def test_model_exported_from_providers() -> None:
    from ecs_agent.providers import Model  # noqa: F401


def test_model_type_exported_from_providers() -> None:
    from ecs_agent.providers import ModelType  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Auto-selection by api_format (no model_type given)
# ---------------------------------------------------------------------------


def test_model_openai_chat_completions_returns_openai_model() -> None:
    from ecs_agent.providers.model_constructor import Model
    from ecs_agent.providers.openai_model import OpenAIModel

    m = Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    assert isinstance(m, OpenAIModel)


def test_model_openai_responses_returns_openai_model() -> None:
    from ecs_agent.providers.model_constructor import Model
    from ecs_agent.providers.openai_model import OpenAIModel

    m = Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_RESPONSES,
    )
    assert isinstance(m, OpenAIModel)


def test_model_anthropic_messages_returns_claude_model() -> None:
    from ecs_agent.providers.claude_model import ClaudeModel
    from ecs_agent.providers.model_constructor import Model

    m = Model(
        "claude-3-5-haiku",
        base_url="https://api.anthropic.com",
        api_key="test-key",
        api_format=ApiFormat.ANTHROPIC_MESSAGES,
    )
    assert isinstance(m, ClaudeModel)


# ---------------------------------------------------------------------------
# 3. Explicit model_type as string
# ---------------------------------------------------------------------------


def test_model_type_openai_returns_openai_model() -> None:
    from ecs_agent.providers.model_constructor import Model
    from ecs_agent.providers.openai_model import OpenAIModel

    m = Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model_type="openai",
    )
    assert isinstance(m, OpenAIModel)


def test_model_type_claude_returns_claude_model() -> None:
    from ecs_agent.providers.claude_model import ClaudeModel
    from ecs_agent.providers.model_constructor import Model

    m = Model(
        "claude-3-5-haiku",
        base_url="https://api.anthropic.com",
        api_key="test-key",
        model_type="claude",
    )
    assert isinstance(m, ClaudeModel)


# ---------------------------------------------------------------------------
# 4. Explicit model_type as class reference
# ---------------------------------------------------------------------------


def test_model_type_as_openai_model_class() -> None:
    from ecs_agent.providers.model_constructor import Model
    from ecs_agent.providers.openai_model import OpenAIModel

    m = Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model_type=OpenAIModel,
    )
    assert isinstance(m, OpenAIModel)


def test_model_type_as_claude_model_class() -> None:
    from ecs_agent.providers.claude_model import ClaudeModel
    from ecs_agent.providers.model_constructor import Model

    m = Model(
        "claude-3-5-haiku",
        base_url="https://api.anthropic.com",
        api_key="test-key",
        model_type=ClaudeModel,
    )
    assert isinstance(m, ClaudeModel)


# ---------------------------------------------------------------------------
# 5. Default api_format inference from model_type
# ---------------------------------------------------------------------------


def test_model_type_openai_defaults_to_chat_completions() -> None:
    from ecs_agent.providers.model_constructor import Model
    from ecs_agent.providers.openai_model import OpenAIModel

    m = Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model_type="openai",
    )
    assert isinstance(m, OpenAIModel)
    assert m._provider_config.api_format == ApiFormat.OPENAI_CHAT_COMPLETIONS


def test_model_type_claude_defaults_to_anthropic_messages() -> None:
    from ecs_agent.providers.claude_model import ClaudeModel
    from ecs_agent.providers.model_constructor import Model

    m = Model(
        "claude-3-5-haiku",
        base_url="https://api.anthropic.com",
        api_key="test-key",
        model_type="claude",
    )
    assert isinstance(m, ClaudeModel)
    assert m._provider_config.api_format == ApiFormat.ANTHROPIC_MESSAGES


# ---------------------------------------------------------------------------
# 6. api_format override when model_type also given
# ---------------------------------------------------------------------------


def test_model_openai_type_with_responses_format() -> None:
    from ecs_agent.providers.model_constructor import Model
    from ecs_agent.providers.openai_model import OpenAIModel

    m = Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model_type="openai",
        api_format=ApiFormat.OPENAI_RESPONSES,
    )
    assert isinstance(m, OpenAIModel)
    assert m._provider_config.api_format == ApiFormat.OPENAI_RESPONSES


# ---------------------------------------------------------------------------
# 7. Conflict detection
# ---------------------------------------------------------------------------


def test_model_conflict_claude_type_with_openai_chat_format_raises() -> None:
    from ecs_agent.providers.model_constructor import Model

    with pytest.raises(ValueError, match="conflict"):
        Model(
            "some-model",
            base_url="https://example.com",
            api_key="test-key",
            model_type="claude",
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        )


def test_model_conflict_claude_type_with_openai_responses_format_raises() -> None:
    from ecs_agent.providers.model_constructor import Model

    with pytest.raises(ValueError, match="conflict"):
        Model(
            "some-model",
            base_url="https://example.com",
            api_key="test-key",
            model_type="claude",
            api_format=ApiFormat.OPENAI_RESPONSES,
        )


def test_model_conflict_openai_type_with_anthropic_format_raises() -> None:
    from ecs_agent.providers.model_constructor import Model

    with pytest.raises(ValueError, match="conflict"):
        Model(
            "some-model",
            base_url="https://example.com",
            api_key="test-key",
            model_type="openai",
            api_format=ApiFormat.ANTHROPIC_MESSAGES,
        )


# ---------------------------------------------------------------------------
# 8. Error: neither api_format nor model_type given
# ---------------------------------------------------------------------------


def test_model_without_api_format_or_model_type_raises() -> None:
    from ecs_agent.providers.model_constructor import Model

    with pytest.raises(ValueError, match="api_format.*model_type|model_type.*api_format"):
        Model(
            "gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )


# ---------------------------------------------------------------------------
# 9. Error: unsupported api_format for auto-selection
# ---------------------------------------------------------------------------


def test_model_embeddings_api_format_raises() -> None:
    from ecs_agent.providers.model_constructor import Model

    with pytest.raises(ValueError):
        Model(
            "text-embedding-3",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            api_format=ApiFormat.OPENAI_EMBEDDINGS,
        )


# ---------------------------------------------------------------------------
# 10. Error: unknown model_type string
# ---------------------------------------------------------------------------


def test_model_unknown_model_type_string_raises() -> None:
    from ecs_agent.providers.model_constructor import Model

    with pytest.raises(ValueError, match="[Uu]nknown model_type"):
        Model(
            "some-model",
            base_url="https://example.com",
            api_key="test-key",
            model_type="unknown_provider_xyz",
        )


# ---------------------------------------------------------------------------
# 11. model_id preserved
# ---------------------------------------------------------------------------


def test_model_id_preserved_openai() -> None:
    from ecs_agent.providers.model_constructor import Model

    m = Model(
        "my-custom-model-42",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    assert m.model_id == "my-custom-model-42"


def test_model_id_preserved_claude() -> None:
    from ecs_agent.providers.model_constructor import Model

    m = Model(
        "claude-special-version-99",
        base_url="https://api.anthropic.com",
        api_key="test-key",
        api_format=ApiFormat.ANTHROPIC_MESSAGES,
    )
    assert m.model_id == "claude-special-version-99"


# ---------------------------------------------------------------------------
# 12. api_format as string
# ---------------------------------------------------------------------------


def test_model_api_format_as_string_openai_chat() -> None:
    from ecs_agent.providers.model_constructor import Model
    from ecs_agent.providers.openai_model import OpenAIModel

    m = Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format="openai_chat_completions",
    )
    assert isinstance(m, OpenAIModel)


def test_model_api_format_as_string_anthropic() -> None:
    from ecs_agent.providers.claude_model import ClaudeModel
    from ecs_agent.providers.model_constructor import Model

    m = Model(
        "claude-3-5-haiku",
        base_url="https://api.anthropic.com",
        api_key="test-key",
        api_format="anthropic_messages",
    )
    assert isinstance(m, ClaudeModel)


def test_model_api_format_invalid_string_raises() -> None:
    from ecs_agent.providers.model_constructor import Model

    with pytest.raises(ValueError):
        Model(
            "some-model",
            base_url="https://example.com",
            api_key="test-key",
            api_format="not_a_valid_format",
        )


# ---------------------------------------------------------------------------
# 13. Returns LLMModel protocol
# ---------------------------------------------------------------------------


def test_model_returns_llm_model_protocol() -> None:
    from ecs_agent.providers.model_constructor import Model
    from ecs_agent.providers.protocol import LLMModel

    m = Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    assert isinstance(m, LLMModel)


# ---------------------------------------------------------------------------
# 14. ModelType enum
# ---------------------------------------------------------------------------


def test_model_type_enum_values() -> None:
    from ecs_agent.providers.model_constructor import ModelType

    assert ModelType.OPENAI == "openai"
    assert ModelType.CLAUDE == "claude"
    assert ModelType.LITELLM == "litellm"


def test_model_with_model_type_enum() -> None:
    from ecs_agent.providers.model_constructor import Model, ModelType
    from ecs_agent.providers.openai_model import OpenAIModel

    m = Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model_type=ModelType.OPENAI,
    )
    assert isinstance(m, OpenAIModel)


# ---------------------------------------------------------------------------
# 15. Extra kwargs forwarded to underlying model
# ---------------------------------------------------------------------------


def test_model_openai_with_extra_kwargs() -> None:
    from ecs_agent.providers.model_constructor import Model
    from ecs_agent.providers.openai_model import OpenAIModel

    # connect_timeout is an accepted kwarg for OpenAIModel
    m = Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        connect_timeout=5.0,
    )
    assert isinstance(m, OpenAIModel)


# ---------------------------------------------------------------------------
# Tests moved from test_model_factory.py (redundant file removed)
# ---------------------------------------------------------------------------


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
