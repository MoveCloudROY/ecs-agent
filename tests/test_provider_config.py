import pytest

from ecs_agent.providers import (
    ApiFormat,
    ModelId,
    ProviderConfig,
    format_model_id,
    parse_model_id,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("openai/gpt-4o", ModelId(provider="openai", model="gpt-4o")),
        (
            "aliyun/qwen3.5-flash",
            ModelId(provider="aliyun", model="qwen3.5-flash"),
        ),
        (
            "anthropic/claude-3-5-sonnet",
            ModelId(provider="anthropic", model="claude-3-5-sonnet"),
        ),
    ],
)
def test_canonical_id_parse_model_id_valid_provider_and_model(
    raw: str, expected: ModelId
) -> None:
    parsed = parse_model_id(raw)

    assert parsed == expected
    assert format_model_id(parsed) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "openai",
        "openai:gpt-4",
        "/foo",
        "openai/",
    ],
)
def test_canonical_id_parse_model_id_rejects_invalid_identifiers(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_model_id(raw)


def test_api_format_enum_has_expected_values() -> None:
    assert ApiFormat.OPENAI_CHAT_COMPLETIONS.value == "openai_chat_completions"
    assert ApiFormat.OPENAI_RESPONSES.value == "openai_responses"
    assert ApiFormat.OPENAI_EMBEDDINGS.value == "openai_embeddings"
    assert ApiFormat.OPENAI_FILES.value == "openai_files"
    assert ApiFormat.ANTHROPIC_MESSAGES.value == "anthropic_messages"


def test_provider_config_construction() -> None:
    config = ProviderConfig(
        provider_id="openai",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        extra_headers={"X-Test": "yes"},
        timeout=30.0,
    )

    assert config.provider_id == "openai"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key == "test-key"
    assert config.api_format is ApiFormat.OPENAI_CHAT_COMPLETIONS
    assert config.extra_headers == {"X-Test": "yes"}
    assert config.timeout == 30.0


def test_provider_config_defaults_for_optional_fields() -> None:
    config = ProviderConfig(
        provider_id="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        api_format=ApiFormat.ANTHROPIC_MESSAGES,
    )

    assert config.extra_headers == {}
    assert config.timeout is None


def test_provider_config_default_headers_are_not_shared() -> None:
    left = ProviderConfig(
        provider_id="openai",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    right = ProviderConfig(
        provider_id="openai",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )

    left.extra_headers["X-Test"] = "left"

    assert right.extra_headers == {}
