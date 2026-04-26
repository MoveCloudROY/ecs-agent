"""Live API tests for the unified Model(...) constructor interface.

Environment variables:
  Anthropic-compatible:
    ANTHROPIC_LIVE_API_KEY (required)
    ANTHROPIC_LIVE_BASE_URL (optional, defaults to https://cc2.caaa.tech)
    ANTHROPIC_LIVE_MODEL (optional, defaults to kimi-for-coding)

  Aliyun / OpenAI-compatible:
    LLM_API_KEY (required)
    ALIYUN_LIVE_CHAT_BASE_URL (optional)
    ALIYUN_LIVE_RESPONSES_BASE_URL (optional)
    ALIYUN_LIVE_MODEL (optional, defaults to qwen3.5-flash)

Tests skip automatically when the required credentials are not present.
"""

from __future__ import annotations

import os

import pytest

from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.model_constructor import Model
from ecs_agent.types import CompletionResult, Message

_ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_LIVE_BASE_URL", "https://cc2.caaa.tech")
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_LIVE_MODEL", "kimi-for-coding")

_ALIYUN_CHAT_BASE_URL = os.getenv(
    "ALIYUN_LIVE_CHAT_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
_ALIYUN_RESPONSES_BASE_URL = (
    os.getenv(
        "ALIYUN_LIVE_RESPONSES_BASE_URL",
        "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    )
)
_ALIYUN_MODEL = os.getenv("ALIYUN_LIVE_MODEL", "qwen3.5-flash")

_GREETING = "Say 'hello world' in exactly those two words and nothing else."


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _complete(model_id: str, base_url: str, api_key: str, api_format: ApiFormat) -> str:
    """Create a Model and get a short text completion. Returns content string."""
    m = Model(model_id, base_url=base_url, api_key=api_key, api_format=api_format)
    result = await m.complete([Message(role="user", content=_GREETING)])
    assert isinstance(result, CompletionResult)
    assert result.message.content
    return result.message.content


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    pytest.skip(f"Live credential '{name}' is not set")


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_anthropic_messages_live() -> None:
    """Model with ANTHROPIC_MESSAGES format against kimi-for-coding compatible endpoint."""
    content = await _complete(
        _ANTHROPIC_MODEL,
        _ANTHROPIC_BASE_URL,
        _require_env("ANTHROPIC_LIVE_API_KEY"),
        ApiFormat.ANTHROPIC_MESSAGES,
    )
    assert len(content.strip()) > 0
    print(f"\n[anthropic_messages] response: {content!r}")


@pytest.mark.asyncio
async def test_model_anthropic_messages_via_model_type_live() -> None:
    """Model with model_type='claude' (no explicit api_format) against anthropic endpoint."""
    m = Model(
        _ANTHROPIC_MODEL,
        base_url=_ANTHROPIC_BASE_URL,
        api_key=_require_env("ANTHROPIC_LIVE_API_KEY"),
        model_type="claude",
    )
    result = await m.complete([Message(role="user", content=_GREETING)])
    assert isinstance(result, CompletionResult)
    assert result.message.content
    print(f"\n[model_type=claude] response: {result.message.content!r}")


@pytest.mark.asyncio
async def test_model_openai_chat_completions_live() -> None:
    """Model with OPENAI_CHAT_COMPLETIONS format against Aliyun/Qwen."""
    content = await _complete(
        _ALIYUN_MODEL,
        _ALIYUN_CHAT_BASE_URL,
        _require_env("LLM_API_KEY"),
        ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    assert len(content.strip()) > 0
    print(f"\n[openai_chat_completions] response: {content!r}")


@pytest.mark.asyncio
async def test_model_openai_chat_completions_via_model_type_live() -> None:
    """Model with model_type='openai' (defaults to OPENAI_CHAT_COMPLETIONS)."""
    m = Model(
        _ALIYUN_MODEL,
        base_url=_ALIYUN_CHAT_BASE_URL,
        api_key=_require_env("LLM_API_KEY"),
        model_type="openai",
    )
    result = await m.complete([Message(role="user", content=_GREETING)])
    assert isinstance(result, CompletionResult)
    assert result.message.content
    print(f"\n[model_type=openai] response: {result.message.content!r}")


@pytest.mark.asyncio
async def test_model_openai_responses_api_live() -> None:
    """Model with OPENAI_RESPONSES format against Aliyun Responses endpoint."""
    import httpx

    try:
        content = await _complete(
            _ALIYUN_MODEL,
            _ALIYUN_RESPONSES_BASE_URL,
            _require_env("LLM_API_KEY"),
            ApiFormat.OPENAI_RESPONSES,
        )
        assert len(content.strip()) > 0
        print(f"\n[openai_responses] response: {content!r}")
    except httpx.ReadTimeout:
        pytest.skip("Aliyun Responses API endpoint timed out (known flaky)")


@pytest.mark.asyncio
async def test_model_api_format_string_live() -> None:
    """Model accepts api_format as a string (e.g. 'openai_chat_completions')."""
    m = Model(
        _ALIYUN_MODEL,
        base_url=_ALIYUN_CHAT_BASE_URL,
        api_key=_require_env("LLM_API_KEY"),
        api_format="openai_chat_completions",
    )
    result = await m.complete([Message(role="user", content=_GREETING)])
    assert isinstance(result, CompletionResult)
    assert result.message.content
    print(f"\n[api_format=string] response: {result.message.content!r}")


@pytest.mark.asyncio
async def test_model_model_id_preserved_after_live_call() -> None:
    """model_id property matches the requested model_id after a real call."""
    model_id = _ALIYUN_MODEL
    m = Model(
        model_id,
        base_url=_ALIYUN_CHAT_BASE_URL,
        api_key=_require_env("LLM_API_KEY"),
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    assert m.model_id == model_id
    result = await m.complete([Message(role="user", content="Hi")])
    assert isinstance(result, CompletionResult)
    assert m.model_id == model_id  # unchanged after call
