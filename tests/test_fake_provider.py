"""Tests for FakeProvider implementation."""

import pytest
from ecs_agent.providers import FakeProvider, LLMProvider
from ecs_agent.types import Message, CompletionResult, ToolCall


@pytest.mark.asyncio
async def test_fake_provider_returns_responses_sequentially() -> None:
    """FakeProvider should return responses in order."""
    resp1 = CompletionResult(message=Message(role="assistant", content="First"))
    resp2 = CompletionResult(message=Message(role="assistant", content="Second"))

    provider = FakeProvider(responses=[resp1, resp2])

    result1 = await provider.complete([Message(role="user", content="Hi")])
    assert result1.message.content == "First"

    result2 = await provider.complete([Message(role="user", content="Hey")])
    assert result2.message.content == "Second"


@pytest.mark.asyncio
async def test_fake_provider_raises_on_exhaustion() -> None:
    """FakeProvider should raise IndexError when responses exhausted."""
    resp = CompletionResult(message=Message(role="assistant", content="Only one"))
    provider = FakeProvider(responses=[resp])

    # First call succeeds
    await provider.complete([Message(role="user", content="1")])

    # Second call should raise
    with pytest.raises((IndexError, StopIteration)):
        await provider.complete([Message(role="user", content="2")])


@pytest.mark.asyncio
async def test_fake_provider_with_empty_list() -> None:
    """FakeProvider with empty list should raise immediately."""
    provider = FakeProvider(responses=[])

    with pytest.raises((IndexError, StopIteration)):
        await provider.complete([Message(role="user", content="Hi")])


@pytest.mark.asyncio
async def test_fake_provider_preserves_tool_calls() -> None:
    """FakeProvider should preserve tool_calls in responses."""
    tool_call = ToolCall(id="tc1", name="search", arguments='{"q":"test"}')
    msg = Message(role="assistant", content="Calling tool", tool_calls=[tool_call])
    resp = CompletionResult(message=msg)

    provider = FakeProvider(responses=[resp])
    result = await provider.complete([Message(role="user", content="Search")])

    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].name == "search"


def test_fake_provider_satisfies_protocol() -> None:
    """FakeProvider should satisfy LLMProvider Protocol structurally."""
    from typing import get_type_hints
    import inspect

    provider = FakeProvider(responses=[])

    # Check that FakeProvider has 'complete' method
    assert hasattr(provider, "complete")

    # Check that complete is async
    assert inspect.iscoroutinefunction(provider.complete)


@pytest.mark.asyncio
async def test_fake_provider_with_usage() -> None:
    """FakeProvider should preserve Usage info in CompletionResult."""
    from ecs_agent.types import Usage

    msg = Message(role="assistant", content="Response")
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    resp = CompletionResult(message=msg, usage=usage)

    provider = FakeProvider(responses=[resp])
    result = await provider.complete([Message(role="user", content="Hi")])

    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.total_tokens == 30
