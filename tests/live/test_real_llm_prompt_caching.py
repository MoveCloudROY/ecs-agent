"""Live LLM scenarios selected explicitly with --live and a model profile."""

from __future__ import annotations


import pytest

from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.types import CompletionResult, Message

from tests.support.models import LiveModelFactory


# A stable system prefix must exceed the model's minimum cacheable size
# (~2048 tokens for Haiku/Sonnet-class models) for a cache write to occur.
_LARGE_STABLE_SYSTEM_PROMPT = (
    "You are a meticulous assistant. Follow these standing instructions.\n\n"
    + "\n".join(
        f"Guideline {i}: Always be precise, cite assumptions, and prefer "
        f"structured answers over prose when enumerating options or steps."
        for i in range(400)
    )
)


def _claude_model(*, live_model_factory: LiveModelFactory) -> ClaudeModel:
    return live_model_factory.claude(max_tokens=64)


@pytest.mark.live(
    formats=["anthropic_messages"], capabilities=["text", "streaming", "cache"]
)
@pytest.mark.cache
@pytest.mark.asyncio
async def test_real_anthropic_prompt_cache_hits_on_second_call(
    *, live_model_factory: LiveModelFactory
) -> None:
    """Two calls sharing a large stable system prefix -> 2nd call reads the cache."""
    model = _claude_model(live_model_factory=live_model_factory)
    messages = [
        Message(role="system", content=_LARGE_STABLE_SYSTEM_PROMPT, cache_control=True),
        Message(role="user", content="Reply with the single word: ok"),
    ]

    # First call primes the cache. Some endpoints report cache_creation here,
    # others only surface cache_read on the warm call, so we don't assert on the
    # cold call's cache fields — only that it succeeded with usage.
    first = await model.complete(messages, stream=False)
    assert isinstance(first, CompletionResult)
    assert first.usage is not None

    second = await model.complete(messages, stream=False)
    assert isinstance(second, CompletionResult)
    assert second.usage is not None
    # Core assertion: the large stable prefix is served from cache on the repeat
    # call (verified live against an Anthropic-format endpoint: ~10k-token prefix
    # -> cache_read_tokens ≈ 9984 on the second call).
    assert (second.usage.cache_read_tokens or 0) > 0, (
        "expected cache_read_tokens > 0 on the second identical-prefix call"
    )
