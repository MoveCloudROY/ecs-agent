from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ecs_agent.providers.claude_provider import ClaudeProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.types import Message, ToolCall, ToolSchema


def test_constructor_stores_configuration() -> None:
    provider = ClaudeProvider(
        api_key="test-key",
        base_url="https://test.anthropic.com",
        model="claude-3-opus-20240229",
        max_tokens=2048,
    )

    assert provider._api_key == "test-key"
    assert provider._base_url == "https://test.anthropic.com"
    assert provider._model == "claude-3-opus-20240229"
    assert provider._max_tokens == 2048


def test_constructor_uses_default_base_url_and_max_tokens() -> None:
    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")

    assert provider._base_url == "https://api.anthropic.com"
    assert provider._max_tokens == 4096


def test_build_messages_extracts_system_and_formats_content_blocks() -> None:
    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    messages = [
        Message(role="system", content="You are concise."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]

    system, anthropic_messages = provider._build_messages(messages)

    assert system == "You are concise."
    assert anthropic_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]


def test_build_messages_converts_tool_result_to_user_tool_result_block() -> None:
    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    messages = [
        Message(role="tool", content="22C and sunny", tool_call_id="toolu_123"),
    ]

    system, anthropic_messages = provider._build_messages(messages)

    assert system is None
    assert anthropic_messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "22C and sunny",
                }
            ],
        }
    ]


def test_build_tools_converts_parameters_to_input_schema() -> None:
    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    tools = [
        ToolSchema(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    ]

    anthropic_tools = provider._build_tools(tools)

    assert anthropic_tools == [
        {
            "name": "get_weather",
            "description": "Get weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


def test_parse_response_text_content_blocks() -> None:
    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    response_data: dict[str, Any] = {
        "content": [{"type": "text", "text": "Hello from Claude"}],
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }

    result = provider._parse_response(response_data)

    assert result.message.role == "assistant"
    assert result.message.content == "Hello from Claude"
    assert result.message.tool_calls is None
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 4
    assert result.usage.total_tokens == 14


def test_parse_response_tool_use_content_blocks() -> None:
    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    response_data: dict[str, Any] = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_456",
                "name": "get_weather",
                "input": {"city": "SF"},
            }
        ]
    }

    result = provider._parse_response(response_data)

    assert result.message.role == "assistant"
    assert result.message.content == ""
    assert result.message.tool_calls is not None
    assert result.message.tool_calls == [
        ToolCall(id="toolu_456", name="get_weather", arguments={"city": "SF"})
    ]


def test_parse_response_mixed_text_and_tool_use_blocks() -> None:
    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    response_data: dict[str, Any] = {
        "content": [
            {"type": "text", "text": "Checking now."},
            {
                "type": "tool_use",
                "id": "toolu_999",
                "name": "lookup",
                "input": {"q": "weather"},
            },
        ]
    }

    result = provider._parse_response(response_data)

    assert result.message.content == "Checking now."
    assert result.message.tool_calls is not None
    assert result.message.tool_calls[0].id == "toolu_999"


@pytest.mark.asyncio
async def test_complete_non_streaming_makes_expected_request() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "content": [{"type": "text", "text": "Response"}],
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = ClaudeProvider(
        api_key="test-key",
        base_url="https://api.anthropic.com",
        model="claude-3-haiku-20240307",
        max_tokens=300,
    )
    provider._client = mock_client

    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Hello"),
        Message(role="tool", content="Sunny", tool_call_id="toolu_111"),
    ]
    tools = [
        ToolSchema(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {}},
        )
    ]

    result = await provider.complete(messages=messages, tools=tools, stream=False)

    assert mock_client.post.called
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://api.anthropic.com/v1/messages"

    assert call_args[1]["headers"] == {
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = call_args[1]["json"]
    assert body["model"] == "claude-3-haiku-20240307"
    assert body["max_tokens"] == 300
    assert body["system"] == "System prompt"
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_111",
                    "content": "Sunny",
                }
            ],
        },
    ]
    assert body["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    assert result.message.content == "Response"


@pytest.mark.asyncio
async def test_complete_raises_on_http_status_error() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.text = "rate limited"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Too Many Requests",
        request=Mock(spec=httpx.Request),
        response=mock_response,
    )

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    provider._client = mock_client

    with pytest.raises(httpx.HTTPStatusError):
        await provider.complete(
            messages=[Message(role="user", content="hello")], stream=False
        )


def test_protocol_compliance() -> None:
    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    assert isinstance(provider, LLMProvider)
