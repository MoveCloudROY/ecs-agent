"""Tests for the shared LLM_API_FORMAT resolver used by live tests."""

import pytest

from ecs_agent.providers.config import ApiFormat
from tests.live.api_format import resolve_live_api_format


def test_resolver_defaults_to_chat_completions_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_API_FORMAT", raising=False)

    assert resolve_live_api_format() is ApiFormat.OPENAI_CHAT_COMPLETIONS


def test_resolver_respects_explicit_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_API_FORMAT", raising=False)

    assert (
        resolve_live_api_format(default=ApiFormat.OPENAI_RESPONSES)
        is ApiFormat.OPENAI_RESPONSES
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("openai_chat_completions", ApiFormat.OPENAI_CHAT_COMPLETIONS),
        ("openai_responses", ApiFormat.OPENAI_RESPONSES),
        ("anthropic_messages", ApiFormat.ANTHROPIC_MESSAGES),
        ("openai", ApiFormat.OPENAI_CHAT_COMPLETIONS),
        ("chat", ApiFormat.OPENAI_CHAT_COMPLETIONS),
        ("responses", ApiFormat.OPENAI_RESPONSES),
        ("anthropic", ApiFormat.ANTHROPIC_MESSAGES),
    ],
)
def test_resolver_accepts_canonical_values_and_aliases(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: ApiFormat
) -> None:
    monkeypatch.setenv("LLM_API_FORMAT", raw)

    assert resolve_live_api_format() is expected


def test_resolver_is_case_and_whitespace_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_FORMAT", "  Anthropic_Messages ")

    assert resolve_live_api_format() is ApiFormat.ANTHROPIC_MESSAGES


def test_resolver_returns_none_for_unrecognized_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_FORMAT", "gemini")

    assert resolve_live_api_format() is None


def test_resolver_treats_blank_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_FORMAT", "   ")

    assert resolve_live_api_format() is ApiFormat.OPENAI_CHAT_COMPLETIONS


def test_transient_live_errors_produce_skip_reason() -> None:
    from tests.live.api_format import live_transient_error_reason

    transient_errors = [
        "ReadTimeout",
        "The read operation timed out",
        "Server error '502 Bad Gateway' for url 'https://gw/v1/chat/completions'",
        "Server error '503 Service Unavailable' for url 'https://gw/v1/chat/completions'",
        "Client error '429 Too Many Requests' for url 'https://gw/v1/chat/completions'",
        "peer closed connection without sending complete message body",
        "All connection attempts failed",
    ]
    for error_text in transient_errors:
        assert live_transient_error_reason(error_text) is not None, error_text


def test_non_transient_live_errors_do_not_skip() -> None:
    from tests.live.api_format import live_transient_error_reason

    hard_failures = [
        "",
        "   ",
        "Client error '401 Unauthorized' for url 'https://gw/v1/chat/completions'",
        "Client error '404 Not Found' for url 'https://gw/v1/chat/completions'",
        "KeyError: 'choices'",
    ]
    for error_text in hard_failures:
        assert live_transient_error_reason(error_text) is None, error_text
