"""Tests for OpenAI-compatible provider."""

import json
import pytest
import httpx
from unittest.mock import AsyncMock, Mock
from ecs_agent.providers.openai_provider import OpenAIProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.types import Message, CompletionResult, ToolSchema, ToolCall, Usage


@pytest.mark.asyncio
async def test_constructor_instantiation() -> None:
    """Test OpenAIProvider can be instantiated with required parameters."""
    provider = OpenAIProvider(
        api_key="test-key", base_url="https://test.openai.com/v1", model="gpt-4o-mini"
    )
    assert provider is not None


@pytest.mark.asyncio
async def test_request_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test HTTP request format matches OpenAI spec."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(
        api_key="test-key", base_url="https://test.openai.com/v1", model="gpt-4o-mini"
    )
    provider._client = mock_client

    messages = [Message(role="user", content="test message")]
    tools = [
        ToolSchema(
            name="test_tool",
            description="test description",
            parameters={"type": "object", "properties": {}},
        )
    ]

    await provider.complete(messages, tools)

    # Verify POST was called
    assert mock_client.post.called
    call_args = mock_client.post.call_args

    # Verify URL
    assert call_args[0][0] == "https://test.openai.com/v1/chat/completions"

    # Verify headers
    assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"
    assert call_args[1]["headers"]["Content-Type"] == "application/json"

    # Verify request body
    body = call_args[1]["json"]
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "test message"}]
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "test description",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.asyncio
async def test_response_parsing_content_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test response parsing extracts content and usage correctly."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(api_key="test-key")
    provider._client = mock_client

    messages = [Message(role="user", content="test")]
    result = await provider.complete(messages)

    assert result.message.role == "assistant"
    assert result.message.content == "Hello world"
    assert result.message.tool_calls is None
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_response_parsing_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test response parsing handles tool calls correctly."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
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
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(api_key="test-key")
    provider._client = mock_client

    messages = [Message(role="user", content="What's the weather?")]
    result = await provider.complete(messages)

    assert result.message.role == "assistant"
    assert result.message.content == ""
    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].id == "call_123"
    assert result.message.tool_calls[0].name == "get_weather"
    assert result.message.tool_calls[0].arguments == '{"location": "NYC"}'


@pytest.mark.asyncio
async def test_http_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test HTTP 4xx/5xx errors are raised as exceptions."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Too Many Requests", request=Mock(spec=httpx.Request), response=mock_response
    )

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(api_key="test-key")
    provider._client = mock_client

    messages = [Message(role="user", content="test")]

    with pytest.raises(httpx.HTTPStatusError):
        await provider.complete(messages)


@pytest.mark.asyncio
async def test_request_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test request format when tools parameter is None."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "response"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(api_key="test-key")
    provider._client = mock_client

    messages = [Message(role="user", content="test")]
    await provider.complete(messages, tools=None)

    # Verify tools field is not included in request when None
    call_args = mock_client.post.call_args
    body = call_args[1]["json"]
    assert "tools" not in body


@pytest.mark.asyncio
async def test_protocol_compliance() -> None:
    """Test OpenAIProvider satisfies LLMProvider Protocol."""
    provider = OpenAIProvider(api_key="test-key")
    assert isinstance(provider, LLMProvider)
