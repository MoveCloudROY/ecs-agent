"""Tests for LLMProvider Protocol."""

import pytest
from typing import Protocol, get_type_hints
from ecs_agent.providers import LLMProvider, OpenAIProvider, FakeProvider
from ecs_agent.types import Message, CompletionResult, ToolSchema, Usage


def test_llmprovider_is_protocol() -> None:
    """LLMProvider should be a Protocol."""
    assert isinstance(LLMProvider, type)
    # Check it's protocol-like by inspecting _is_protocol
    assert hasattr(LLMProvider, "_is_protocol")


def test_llmprovider_has_complete_method() -> None:
    """LLMProvider should have a complete method."""
    assert hasattr(LLMProvider, "complete")


def test_llmprovider_complete_signature() -> None:
    """complete method should have correct signature."""
    # Get the method
    complete_method = getattr(LLMProvider, "complete")
    assert complete_method is not None

    # Check it's callable
    assert callable(complete_method)

    # Get type hints
    hints = get_type_hints(complete_method)

    # Should have 'messages' parameter
    assert "messages" in hints

    # Should have 'tools' parameter
    assert "tools" in hints

    # Should have 'return' annotation
    assert "return" in hints
    # Return type should include CompletionResult and AsyncIterator[StreamDelta]


def test_llmprovider_complete_is_async() -> None:
    """complete method should be async."""
    import inspect

    complete_method = getattr(LLMProvider, "complete")
    assert inspect.iscoroutinefunction(complete_method)


def test_llmprovider_has_stream_parameter() -> None:
    """complete method should have stream parameter."""
    import inspect

    complete_method = getattr(LLMProvider, "complete")
    sig = inspect.signature(complete_method)
    assert "stream" in sig.parameters
    assert sig.parameters["stream"].default is False


def test_llmprovider_has_response_format_parameter() -> None:
    """complete method should have response_format parameter."""
    import inspect

    complete_method = getattr(LLMProvider, "complete")
    sig = inspect.signature(complete_method)
    assert "response_format" in sig.parameters
    assert sig.parameters["response_format"].default is None


def test_openai_provider_conforms_to_protocol() -> None:
    """OpenAIProvider should conform to LLMProvider Protocol."""
    assert issubclass(OpenAIProvider, LLMProvider)


def test_fake_provider_conforms_to_protocol() -> None:
    """FakeProvider should conform to LLMProvider Protocol."""
    assert issubclass(FakeProvider, LLMProvider)


@pytest.mark.asyncio
async def test_fake_provider_backward_compatibility() -> None:
    """FakeProvider should work with old-style calls (no stream/response_format)."""
    message = Message(role="assistant", content="test response")
    result = CompletionResult(message=message, usage=None)
    provider = FakeProvider([result])

    # Old-style call without new parameters should work
    response = await provider.complete([Message(role="user", content="test")])
    assert response.message.content == "test response"


@pytest.mark.asyncio
async def test_fake_provider_new_params() -> None:
    """FakeProvider should accept new stream and response_format params."""
    message = Message(role="assistant", content="test response")
    result = CompletionResult(message=message, usage=None)
    provider = FakeProvider([result])

    # New-style call with new parameters should work
    response = await provider.complete(
        [Message(role="user", content="test")],
        stream=False,
        response_format=None,
    )
    assert response.message.content == "test response"