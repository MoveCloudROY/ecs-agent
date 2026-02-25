import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ecs_agent.providers.claude_provider import ClaudeProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.types import Message, StreamDelta, ToolCall, ToolSchema


class _MockStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.raise_for_status = Mock()

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class _MockStreamContext:
    def __init__(self, response: _MockStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _MockStreamResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _anthropic_sse(event_name: str, payload: dict[str, Any]) -> list[str]:
    return [f"event: {event_name}", f"data: {json.dumps(payload)}", ""]


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


@pytest.mark.asyncio
async def test_complete_streaming_returns_async_iterator() -> None:
    lines: list[str] = []
    lines.extend(
        _anthropic_sse(
            "message_start",
            {"type": "message_start", "message": {"id": "msg_1", "type": "message"}},
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"},
            },
        )
    )
    lines.extend(
        _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    )
    lines.extend(_anthropic_sse("message_stop", {"type": "message_stop"}))
    stream_response = _MockStreamResponse(lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    provider._client = mock_client

    stream_iter = await provider.complete(
        messages=[Message(role="user", content="hello")],
        stream=True,
    )

    assert isinstance(stream_iter, AsyncIterator)
    deltas = [delta async for delta in stream_iter]
    assert len(deltas) == 1
    assert deltas[0] == StreamDelta(content="Hi")


@pytest.mark.asyncio
async def test_complete_streaming_uses_stream_request_with_stream_true() -> None:
    lines: list[str] = []
    lines.extend(_anthropic_sse("message_start", {"type": "message_start"}))
    lines.extend(_anthropic_sse("message_stop", {"type": "message_stop"}))
    stream_response = _MockStreamResponse(lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    provider = ClaudeProvider(
        api_key="test-key",
        base_url="https://api.anthropic.com",
        model="claude-3-haiku-20240307",
    )
    provider._client = mock_client

    stream_iter = await provider.complete(
        messages=[Message(role="user", content="hello")],
        stream=True,
    )
    _ = [delta async for delta in stream_iter]

    assert not mock_client.post.called
    stream_call = mock_client.stream.call_args
    assert stream_call[0][0] == "POST"
    assert stream_call[0][1] == "https://api.anthropic.com/v1/messages"
    assert stream_call[1]["json"]["stream"] is True


@pytest.mark.asyncio
async def test_streaming_yields_text_deltas_from_content_block_delta() -> None:
    lines: list[str] = []
    lines.extend(_anthropic_sse("message_start", {"type": "message_start"}))
    lines.extend(
        _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world"},
            },
        )
    )
    lines.extend(
        _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    )
    lines.extend(_anthropic_sse("message_stop", {"type": "message_stop"}))
    stream_response = _MockStreamResponse(lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    provider._client = mock_client

    stream_iter = await provider.complete(
        messages=[Message(role="user", content="hello")],
        stream=True,
    )
    deltas = [delta async for delta in stream_iter]

    assert [delta.content for delta in deltas] == ["Hello", " world"]


@pytest.mark.asyncio
async def test_streaming_accumulates_tool_use_input_json_until_content_block_stop() -> (
    None
):
    lines: list[str] = []
    lines.extend(_anthropic_sse("message_start", {"type": "message_start"}))
    lines.extend(
        _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {},
                },
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"city":"'},
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": 'Paris"}'},
            },
        )
    )
    lines.extend(
        _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 1})
    )
    lines.extend(_anthropic_sse("message_stop", {"type": "message_stop"}))
    stream_response = _MockStreamResponse(lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    provider._client = mock_client

    stream_iter = await provider.complete(
        messages=[Message(role="user", content="weather")],
        stream=True,
    )
    deltas = [delta async for delta in stream_iter]

    assert len(deltas) == 1
    assert deltas[0].tool_calls is not None
    assert deltas[0].tool_calls == [
        ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Paris"})
    ]


@pytest.mark.asyncio
async def test_streaming_tool_use_invalid_json_yields_partial_arguments() -> None:
    lines: list[str] = []
    lines.extend(_anthropic_sse("message_start", {"type": "message_start"}))
    lines.extend(
        _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "lookup",
                    "input": {},
                },
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":"news"'},
            },
        )
    )
    lines.extend(
        _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    )
    lines.extend(_anthropic_sse("message_stop", {"type": "message_stop"}))
    stream_response = _MockStreamResponse(lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    provider._client = mock_client

    stream_iter = await provider.complete(
        messages=[Message(role="user", content="lookup")],
        stream=True,
    )
    deltas = [delta async for delta in stream_iter]

    assert len(deltas) == 1
    assert deltas[0].tool_calls is not None
    assert deltas[0].tool_calls[0].arguments == {"_partial": '{"q":"news"'}


@pytest.mark.asyncio
async def test_streaming_handles_message_delta_finish_reason() -> None:
    lines: list[str] = []
    lines.extend(_anthropic_sse("message_start", {"type": "message_start"}))
    lines.extend(
        _anthropic_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 22},
            },
        )
    )
    lines.extend(_anthropic_sse("message_stop", {"type": "message_stop"}))
    stream_response = _MockStreamResponse(lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    provider._client = mock_client

    stream_iter = await provider.complete(
        messages=[Message(role="user", content="x")],
        stream=True,
    )
    deltas = [delta async for delta in stream_iter]

    assert len(deltas) == 1
    assert deltas[0].finish_reason == "tool_use"


@pytest.mark.asyncio
async def test_streaming_skips_empty_lines_and_done_marker() -> None:
    lines: list[str] = ["", "data: [DONE]"]
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ignored"},
            },
        )
    )
    stream_response = _MockStreamResponse(lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku-20240307")
    provider._client = mock_client

    stream_iter = await provider.complete(
        messages=[Message(role="user", content="x")],
        stream=True,
    )
    deltas = [delta async for delta in stream_iter]

    assert deltas == []


@pytest.mark.asyncio
async def test_streaming_timeout_uses_read_none_and_custom_values() -> None:
    lines: list[str] = []
    lines.extend(_anthropic_sse("message_start", {"type": "message_start"}))
    lines.extend(_anthropic_sse("message_stop", {"type": "message_stop"}))
    stream_response = _MockStreamResponse(lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    provider = ClaudeProvider(
        api_key="test-key",
        connect_timeout=5.0,
        read_timeout=90.0,
        write_timeout=8.0,
        pool_timeout=6.0,
    )
    provider._client = mock_client

    stream_iter = await provider.complete(
        messages=[Message(role="user", content="hello")],
        stream=True,
    )
    _ = [delta async for delta in stream_iter]

    timeout = mock_client.stream.call_args[1]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 5.0
    assert timeout.read is None
    assert timeout.write == 8.0
    assert timeout.pool == 6.0
