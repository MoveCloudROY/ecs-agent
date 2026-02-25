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
    import ecs_agent.providers.litellm_provider

    monkeypatch.setattr("ecs_agent.providers.litellm_provider.HAS_LITELLM", False)

    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    with pytest.raises(ImportError) as exc_info:
        LiteLLMProvider(model="openai/gpt-4", api_key="test")

    assert "litellm is required" in str(exc_info.value)
    assert "pip install litellm" in str(exc_info.value)


@pytest.mark.asyncio
async def test_complete_non_streaming(mock_litellm) -> None:
    """Test non-streaming completion delegates to litellm.acompletion correctly."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    # Mock response in OpenAI format
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello from litellm!",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    mock_litellm.acompletion.return_value = mock_response

    provider = LiteLLMProvider(model="openai/gpt-4", api_key="test-key")
    messages = [Message(role="user", content="Hello")]

    result = await provider.complete(messages, stream=False)

    assert isinstance(result, CompletionResult)
    assert result.message.role == "assistant"
    assert result.message.content == "Hello from litellm!"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15

    # Verify litellm.acompletion was called with correct args
    assert mock_litellm.acompletion.called
    call_kwargs = mock_litellm.acompletion.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-4"
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert call_kwargs["stream"] is False


@pytest.mark.asyncio
async def test_complete_with_tools(mock_litellm) -> None:
    """Test tool schemas are passed correctly to litellm."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
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
    mock_litellm.acompletion.return_value = mock_response

    provider = LiteLLMProvider(model="anthropic/claude-3-opus-20240229", api_key="test")
    messages = [Message(role="user", content="What's the weather?")]
    tools = [
        ToolSchema(
            name="get_weather",
            description="Get current weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        )
    ]

    result = await provider.complete(messages, tools=tools, stream=False)

    assert isinstance(result, CompletionResult)
    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].id == "call_123"
    assert result.message.tool_calls[0].name == "get_weather"
    assert result.message.tool_calls[0].arguments == {"location": "NYC"}

    # Verify tools were passed in OpenAI format
    call_kwargs = mock_litellm.acompletion.call_args[1]
    assert "tools" in call_kwargs
    assert call_kwargs["tools"][0]["type"] == "function"
    assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_streaming_complete(mock_litellm) -> None:
    """Test streaming completion yields StreamDelta objects."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider
    from ecs_agent.types import StreamDelta

    # Mock streaming response (async iterator)
    async def mock_stream():
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
        ]
        for chunk in chunks:
            yield chunk

    mock_litellm.acompletion.return_value = mock_stream()

    provider = LiteLLMProvider(model="openai/gpt-4", api_key="test")
    messages = [Message(role="user", content="Hi")]

    result = await provider.complete(messages, stream=True)

    # Should return AsyncIterator
    assert hasattr(result, "__aiter__")

    deltas = []
    async for delta in result:
        assert isinstance(delta, StreamDelta)
        deltas.append(delta)

    assert len(deltas) == 3
    assert deltas[0].content == "Hello"
    assert deltas[1].content == " world"
    assert deltas[2].finish_reason == "stop"
    assert deltas[2].usage is not None
    assert deltas[2].usage.total_tokens == 8

    # Verify stream=True was passed
    call_kwargs = mock_litellm.acompletion.call_args[1]
    assert call_kwargs["stream"] is True


@pytest.mark.asyncio
async def test_protocol_compliance(mock_litellm) -> None:
    """Test LiteLLMProvider satisfies LLMProvider Protocol."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(model="openai/gpt-4", api_key="test")
    assert isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_response_format_passed_through(mock_litellm) -> None:
    """Test response_format parameter is passed to litellm."""
    from ecs_agent.providers.litellm_provider import LiteLLMProvider

    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": '{"result": "ok"}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_litellm.acompletion.return_value = mock_response

    provider = LiteLLMProvider(model="openai/gpt-4", api_key="test")
    messages = [Message(role="user", content="Give me JSON")]
    response_format = {"type": "json_object"}

    await provider.complete(messages, response_format=response_format, stream=False)

    call_kwargs = mock_litellm.acompletion.call_args[1]
    assert call_kwargs["response_format"] == {"type": "json_object"}
