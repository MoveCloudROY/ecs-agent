from pathlib import Path

import pytest

from ecs_agent.providers import ApiFormat
from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.config import ProviderEntry
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.providers.registry import ProviderRegistry, get_model


def test_provider_entry_construction_defaults_and_repr_hides_api_key() -> None:
    entry = ProviderEntry(
        base_url="https://api.example.com/v1",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        api_key="secret-key",
    )

    assert entry.base_url == "https://api.example.com/v1"
    assert entry.api_format is ApiFormat.OPENAI_CHAT_COMPLETIONS
    assert entry.api_key == "secret-key"
    assert entry.api_key_env is None
    assert entry.extra_headers == {}
    assert entry.timeout is None
    assert entry.default_max_tokens == 4096
    assert "secret-key" not in repr(entry)


def test_provider_registry_from_dict_happy_path_and_invalid_api_format() -> None:
    registry = ProviderRegistry.from_dict(
        {
            "aliyun": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
                "api_format": "openai_chat_completions",
                "api_key_env": "LLM_API_KEY",
            }
        }
    )

    entry = registry.get_entry("aliyun")
    assert entry.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert entry.api_format is ApiFormat.OPENAI_CHAT_COMPLETIONS
    assert entry.api_key_env == "LLM_API_KEY"

    with pytest.raises(ValueError):
        ProviderRegistry.from_dict(
            {
                "broken": {
                    "base_url": "https://example.com",
                    "api_format": "not_a_real_format",
                }
            }
        )


def test_provider_registry_get_entry_and_missing_provider() -> None:
    registry = ProviderRegistry.from_dict(
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_format": "openai_chat_completions",
                "api_key": "k",
            }
        }
    )

    entry = registry.get_entry("openai")
    assert entry.api_key == "k"

    with pytest.raises(KeyError):
        registry.get_entry("missing")


def test_provider_registry_from_toml(tmp_path: Path) -> None:
    toml_file = tmp_path / "providers.toml"
    toml_file.write_text(
        """
[providers.aliyun]
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
api_format = "openai_responses"
api_key_env = "LLM_API_KEY"
timeout = 30.0
default_max_tokens = 1234
""".strip(),
        encoding="utf-8",
    )

    registry = ProviderRegistry.from_toml(toml_file)
    entry = registry.get_entry("aliyun")

    assert entry.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert entry.api_format is ApiFormat.OPENAI_RESPONSES
    assert entry.api_key_env == "LLM_API_KEY"
    assert entry.timeout == 30.0
    assert entry.default_max_tokens == 1234


def test_get_model_explicit_api_key_returns_openai_provider() -> None:
    registry = ProviderRegistry.from_dict(
        {
            "aliyun": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_format": "openai_chat_completions",
            }
        }
    )

    provider = get_model(
        "aliyun/qwen3.5-flash",
        registry=registry,
        api_key="explicit-key",
    )

    assert isinstance(provider, OpenAIModel)


def test_get_model_resolves_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    registry = ProviderRegistry.from_dict(
        {
            "aliyun": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_format": "openai_chat_completions",
                "api_key_env": "LLM_API_KEY",
            }
        }
    )

    provider = get_model("aliyun/qwen3.5-flash", registry=registry)

    assert isinstance(provider, OpenAIModel)


def test_get_model_openai_responses_returns_openai_provider() -> None:
    registry = ProviderRegistry.from_dict(
        {
            "aliyun-responses": {
                "base_url": "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
                "api_format": "openai_responses",
                "api_key": "test-key",
            }
        }
    )

    provider = get_model("aliyun-responses/qwen3.5-flash", registry=registry)

    assert isinstance(provider, OpenAIModel)


def test_get_model_anthropic_messages_returns_claude_provider() -> None:
    registry = ProviderRegistry.from_dict(
        {
            "aliyun-anthropic": {
                "base_url": "https://dashscope.aliyuncs.com/apps/anthropic",
                "api_format": "anthropic_messages",
                "api_key": "test-key",
                "default_max_tokens": 8192,
            }
        }
    )

    provider = get_model("aliyun-anthropic/kimi-k2.5", registry=registry)

    assert isinstance(provider, ClaudeModel)


def test_get_model_embeddings_format_raises_value_error() -> None:
    registry = ProviderRegistry.from_dict(
        {
            "embed": {
                "base_url": "https://api.openai.com/v1",
                "api_format": "openai_embeddings",
                "api_key": "test-key",
            }
        }
    )

    with pytest.raises(ValueError):
        get_model("embed/text-embedding-3-small", registry=registry)


def test_get_model_missing_provider_raises_key_error() -> None:
    registry = ProviderRegistry.from_dict(
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_format": "openai_chat_completions",
                "api_key": "test-key",
            }
        }
    )

    with pytest.raises(KeyError):
        get_model("missing/gpt-4o-mini", registry=registry)


def test_get_model_no_api_key_resolvable_raises_value_error() -> None:
    registry = ProviderRegistry.from_dict(
        {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_format": "openai_chat_completions",
            }
        }
    )

    with pytest.raises(ValueError):
        get_model("openai/gpt-4o-mini", registry=registry)
