"""Tests for LiteLLM provider with mocked dependencies."""

import json
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from typing import AsyncIterator
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.types import Message, CompletionResult, ToolSchema, ToolCall, Usage


@pytest.fixture
def mock_litellm(monkeypatch: pytest.MonkeyPatch):
    """Mock the litellm module so tests pass without litellm installed."""
    # Create a mock module with acompletion function
    mock_module = MagicMock()
    mock_acompletion = AsyncMock()
    mock_module.acompletion = mock_acompletion

    # Inject mock before reload
    monkeypatch.setitem(__import__("sys").modules, "litellm", mock_module)

    # Reload the provider module so it picks up the mocked litellm from sys.modules
    import importlib
    import ecs_agent.providers.litellm_provider

    importlib.reload(ecs_agent.providers.litellm_provider)

    # Make HAS_LITELLM=True so provider can be instantiated
    monkeypatch.setattr("ecs_agent.providers.litellm_provider.HAS_LITELLM", True)
    return mock_module


@pytest.mark.asyncio
async def test_constructor_instantiation(mock_litellm) -> None:
    """Test LiteLLMProvider can be instantiated with required parameters."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(
        model="anthropic/claude-3-opus-20240229",
        api_key="test-key",
        base_url="https://test.api.com/v1",
    )
    assert provider is not None


@pytest.mark.asyncio
async def test_import_guard_raises_when_litellm_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test constructor raises ImportError with helpful message when litellm not available."""
    # Force HAS_LITELLM=False
    monkeypatch.setattr("ecs_agent.providers.litellm_provider.HAS_LITELLM", False)

    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    with pytest.raises(ImportError, match="litellm"):
        LiteLLMProvider(model="gpt-4o", api_key="test-key")


@pytest.mark.asyncio
async def test_protocol_compliance(mock_litellm) -> None:
    """Test LiteLLMProvider satisfies LLMProvider Protocol."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(model="gpt-4o", api_key="test-key")
    assert isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_complete_returns_completion_result(mock_litellm) -> None:
    """Test complete() returns CompletionResult with message and usage."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    mock_litellm.acompletion.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    provider = LiteLLMProvider(model="gpt-4o", api_key="test-key")
    messages = [Message(role="user", content="test message")]
    result = await provider.complete(messages)

    assert isinstance(result, CompletionResult)
    assert result.message.role == "assistant"
    assert result.message.content == "test response"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_complete_handles_tool_calls(mock_litellm) -> None:
    """Test complete() parses tool calls correctly."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    mock_litellm.acompletion.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }

    provider = LiteLLMProvider(model="gpt-4o", api_key="test-key")
    messages = [Message(role="user", content="What's the weather?")]
    result = await provider.complete(messages)

    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].id == "call_123"
    assert result.message.tool_calls[0].name == "get_weather"
    assert result.message.tool_calls[0].arguments == {"location": "NYC"}


@pytest.mark.asyncio
async def test_stream_yields_deltas(mock_litellm) -> None:
    """Test stream() yields StreamDelta objects."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider
    from ecs_agent.types import StreamDelta

    async def mock_stream():
        yield {
            "choices": [{"delta": {"role": "assistant", "content": "Hello"}}],
            "usage": None,
        }
        yield {
            "choices": [{"delta": {"content": " world"}}],
            "usage": None,
        }
        yield {
            "choices": [{"delta": {}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

    mock_litellm.acompletion.return_value = mock_stream()

    provider = LiteLLMProvider(model="gpt-4o", api_key="test-key")
    messages = [Message(role="user", content="test")]
    deltas = []
    async for delta in provider.stream(messages):
        deltas.append(delta)

    assert len(deltas) == 3
    assert deltas[0].content == "Hello"
    assert deltas[1].content == " world"
    assert deltas[2].usage is not None
    assert deltas[2].usage.total_tokens == 7


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_convert_messages_serializes_tool_call_arguments_as_json_string(mock_litellm) -> None:
    """Test _convert_messages_to_openai serializes tool call arguments as JSON string, not dict."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(
        model="anthropic/claude-3-opus-20240229",
        api_key="test-key",
    )

    # Create message with tool call containing dict arguments
    message = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id="call_123",
                name="test_function",
                arguments={"key": "value", "count": 42},
            )
        ],
    )

    # Convert to OpenAI format
    result = provider._convert_messages_to_openai([message])

    # Extract arguments field
    args = result[0]["tool_calls"][0]["function"]["arguments"]

    # CRITICAL: Must be JSON string, not dict
    assert isinstance(args, str), f"Expected string, got {type(args)}"

    # Verify round-trip: string should deserialize back to original dict
    assert json.loads(args) == {"key": "value", "count": 42}