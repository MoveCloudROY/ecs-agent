import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.types import (
    FileRefPart,
    ImageUrlPart,
    Message,
    StreamDelta,
    ToolCall,
    ToolSchema,
)

COMPACTION_SENTINEL = "[COMPACTION SUMMARY]\n"


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


def _anthropic_config(
    *,
    api_key: str = "test-key",
    base_url: str = "https://api.anthropic.com",
    enable_prompt_caching: bool = False,
) -> ProviderConfig:
    # Prompt caching defaults OFF here so message/tool-conversion tests assert the
    # plain wire shape. Caching behaviour has its own dedicated tests below.
    return ProviderConfig(
        provider_id="anthropic",
        base_url=base_url,
        api_key=api_key,
        api_format=ApiFormat.ANTHROPIC_MESSAGES,
        enable_prompt_caching=enable_prompt_caching,
    )


def test_constructor_stores_configuration() -> None:
    model = ClaudeModel(
        config=_anthropic_config(
            api_key="test-key",
            base_url="https://test.anthropic.com",
        ),
        model="claude-3-opus-20240229",
        max_tokens=2048,
    )

    assert model._api_key == "test-key"
    assert model._base_url == "https://test.anthropic.com"
    assert model._model == "claude-3-opus-20240229"
    assert model._max_tokens == 2048


def test_constructor_uses_default_base_url_and_max_tokens() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )

    assert model._base_url == "https://api.anthropic.com"
    assert model._max_tokens == 65535


def test_build_messages_extracts_system_and_formats_content_blocks() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    messages = [
        Message(role="system", content="You are concise."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]

    system, anthropic_messages = model._build_messages(messages)

    assert system == "You are concise."
    assert anthropic_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]


def test_build_messages_converts_tool_result_to_user_tool_result_block() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    messages = [
        Message(role="tool", content="22C and sunny", tool_call_id="toolu_123"),
    ]

    system, anthropic_messages = model._build_messages(messages)

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


def test_build_messages_replays_assistant_thinking_before_tool_use() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="get_weather",
                    arguments={"city": "San Francisco"},
                )
            ],
            reasoning_content="Need the weather data before answering.",
            reasoning_signature="sig_123",
        ),
        Message(role="tool", content="22C and sunny", tool_call_id="call_123"),
    ]

    system, anthropic_messages = model._build_messages(messages)

    assert system is None
    assert anthropic_messages == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Need the weather data before answering.",
                    "signature": "sig_123",
                },
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                    "input": {"city": "San Francisco"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_123",
                    "content": "22C and sunny",
                }
            ],
        },
    ]


def test_build_messages_groups_parallel_tool_results_after_assistant_tool_use() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="get_weather",
                    arguments={"city": "San Francisco"},
                ),
                ToolCall(
                    id="call_456",
                    name="get_time",
                    arguments={"timezone": "America/Los_Angeles"},
                ),
            ],
        ),
        Message(role="tool", content="22C and sunny", tool_call_id="call_123"),
        Message(role="tool", content="10:30 AM", tool_call_id="call_456"),
    ]

    system, anthropic_messages = model._build_messages(messages)

    assert system is None
    assert anthropic_messages == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                    "input": {"city": "San Francisco"},
                },
                {
                    "type": "tool_use",
                    "id": "call_456",
                    "name": "get_time",
                    "input": {"timezone": "America/Los_Angeles"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_123",
                    "content": "22C and sunny",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_456",
                    "content": "10:30 AM",
                },
            ],
        },
    ]


def test_build_messages_delivers_summary_xml_in_system_string() -> None:
    """Test that Anthropic adapter delivers XML summary in system string."""
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )

    summary_xml = "<chat_history_summary>Summary: X happened</chat_history_summary>"
    system, anthropic_messages = model._build_messages(
        [
            Message(role="system", content=f"You are helpful.\n\n{summary_xml}"),
            Message(role="user", content="Hello"),
        ]
    )

    assert system is not None
    assert summary_xml in system
    assert len(anthropic_messages) >= 1
    assert anthropic_messages[-1]["role"] == "user"


def test_build_tools_converts_parameters_to_input_schema() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
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

    anthropic_tools = model._build_tools(tools)

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
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    response_data: dict[str, Any] = {
        "content": [{"type": "text", "text": "Hello from Claude"}],
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }

    result = model._parse_response(response_data)

    assert result.message.role == "assistant"
    assert result.message.content == "Hello from Claude"
    assert result.message.tool_calls is None
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 4
    assert result.usage.total_tokens == 14


def test_parse_response_usage_includes_cache_fields_and_metadata() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    response_data: dict[str, Any] = {
        "content": [{"type": "text", "text": "Hello from Claude"}],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        },
    }

    result = model._parse_response(response_data)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 4
    assert result.usage.total_tokens == 14
    assert result.usage.cache_creation_tokens == 3
    assert result.usage.cache_read_tokens == 2
    assert result.usage.provider_id == "anthropic"
    assert result.usage.model == "claude-3-haiku-20240307"


@pytest.mark.parametrize(
    ("parts", "expected_part_name"),
    [
        ([ImageUrlPart(url="https://example.com/image.png")], "ImageUrlPart"),
        ([FileRefPart(file_id="file_123", filename="report.pdf")], "FileRefPart"),
    ],
)
def test_build_messages_unsupported_multimodal_without_vision_raises(
    parts: list[ImageUrlPart | FileRefPart],
    expected_part_name: str,
) -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported multimodal part for Anthropic messages endpoint: "
            f"{expected_part_name}"
        ),
    ):
        model._build_messages([Message(role="user", content="", parts=parts)])


def test_parse_response_tool_use_content_blocks() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
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

    result = model._parse_response(response_data)

    assert result.message.role == "assistant"
    assert result.message.content == ""
    assert result.message.tool_calls is not None
    assert result.message.tool_calls == [
        ToolCall(id="toolu_456", name="get_weather", arguments={"city": "SF"})
    ]


def test_parse_response_preserves_thinking_block_for_replay() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    response_data: dict[str, Any] = {
        "content": [
            {
                "type": "thinking",
                "thinking": "Need the weather tool result first.",
                "signature": "sig_456",
            },
            {
                "type": "tool_use",
                "id": "toolu_456",
                "name": "get_weather",
                "input": {"city": "SF"},
            },
        ]
    }

    result = model._parse_response(response_data)

    assert result.reasoning_content == "Need the weather tool result first."
    assert result.message.reasoning_content == "Need the weather tool result first."
    assert result.message.reasoning_signature == "sig_456"
    assert result.message.tool_calls == [
        ToolCall(id="toolu_456", name="get_weather", arguments={"city": "SF"})
    ]


def test_parse_response_preserves_empty_thinking_block_for_replay() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    response_data: dict[str, Any] = {
        "content": [
            {
                "type": "thinking",
                "thinking": "",
                "signature": "sig_empty",
            },
            {
                "type": "tool_use",
                "id": "toolu_empty",
                "name": "lookup",
                "input": {"q": "weather"},
            },
        ]
    }

    result = model._parse_response(response_data)

    assert result.reasoning_content == ""
    assert result.message.reasoning_content == ""
    assert result.message.reasoning_signature == "sig_empty"


def test_parse_response_mixed_text_and_tool_use_blocks() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
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

    result = model._parse_response(response_data)

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

    model = ClaudeModel(
        config=_anthropic_config(
            api_key="test-key", base_url="https://api.anthropic.com"
        ),
        model="claude-3-haiku-20240307",
        max_tokens=300,
    )
    model._client = mock_client

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

    result = await model.complete(messages=messages, tools=tools, stream=False)

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

    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    model._client = mock_client

    with pytest.raises(httpx.HTTPStatusError):
        await model.complete(
            messages=[Message(role="user", content="hello")], stream=False
        )


def test_protocol_compliance() -> None:
    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    assert isinstance(model, LLMModel)


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

    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    model._client = mock_client

    stream_iter = await model.complete(
        messages=[Message(role="user", content="hello")],
        stream=True,
    )

    assert isinstance(stream_iter, AsyncIterator)
    deltas = [delta async for delta in stream_iter]
    assert len(deltas) == 1
    assert deltas[0] == StreamDelta(content="Hi")


@pytest.mark.asyncio
async def test_streaming_yields_thinking_deltas_signature_and_usage() -> None:
    """Streaming parity with the non-streaming path for extended thinking.

    Regression: the streaming handler used to drop thinking_delta /
    signature_delta events and never surfaced usage, so streamed Claude calls
    reported zero tokens and lost reasoning content.
    """
    lines: list[str] = []
    lines.extend(
        _anthropic_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "usage": {
                        "input_tokens": 25,
                        "cache_creation_input_tokens": 5,
                        "cache_read_input_tokens": 10,
                        "output_tokens": 1,
                    },
                },
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me "},
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "think"},
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig-abc"},
            },
        )
    )
    lines.extend(
        _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    )
    lines.extend(
        _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        )
    )
    lines.extend(
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Answer"},
            },
        )
    )
    lines.extend(
        _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 1})
    )
    lines.extend(
        _anthropic_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 40},
            },
        )
    )
    lines.extend(_anthropic_sse("message_stop", {"type": "message_stop"}))
    stream_response = _MockStreamResponse(lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    model._client = mock_client

    stream_iter = await model.complete(
        messages=[Message(role="user", content="hello")],
        stream=True,
    )
    deltas = [delta async for delta in stream_iter]

    reasoning_deltas = [
        delta.reasoning_content
        for delta in deltas
        if delta.reasoning_content is not None
    ]
    assert reasoning_deltas == ["Let me ", "think"]

    signature_deltas = [
        delta.reasoning_signature
        for delta in deltas
        if delta.reasoning_signature is not None
    ]
    assert signature_deltas == ["sig-abc"]

    assert [delta.content for delta in deltas if delta.content is not None] == [
        "Answer"
    ]

    final_deltas = [delta for delta in deltas if delta.finish_reason is not None]
    assert len(final_deltas) == 1
    final = final_deltas[0]
    assert final.finish_reason == "end_turn"
    assert final.usage is not None
    assert final.usage.prompt_tokens == 25
    assert final.usage.completion_tokens == 40
    assert final.usage.total_tokens == 65
    assert final.usage.cache_creation_tokens == 5
    assert final.usage.cache_read_tokens == 10
    assert final.usage.provider_id == "anthropic"
    assert final.usage.model == "claude-3-haiku-20240307"


@pytest.mark.asyncio
async def test_complete_streaming_uses_stream_request_with_stream_true() -> None:
    lines: list[str] = []
    lines.extend(_anthropic_sse("message_start", {"type": "message_start"}))
    lines.extend(_anthropic_sse("message_stop", {"type": "message_stop"}))
    stream_response = _MockStreamResponse(lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = ClaudeModel(
        config=_anthropic_config(
            api_key="test-key", base_url="https://api.anthropic.com"
        ),
        model="claude-3-haiku-20240307",
    )
    model._client = mock_client

    stream_iter = await model.complete(
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

    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    model._client = mock_client

    stream_iter = await model.complete(
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

    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    model._client = mock_client

    stream_iter = await model.complete(
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

    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    model._client = mock_client

    stream_iter = await model.complete(
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

    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    model._client = mock_client

    stream_iter = await model.complete(
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

    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"), model="claude-3-haiku-20240307"
    )
    model._client = mock_client

    stream_iter = await model.complete(
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

    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"),
        connect_timeout=5.0,
        read_timeout=90.0,
        write_timeout=8.0,
        pool_timeout=6.0,
    )
    model._client = mock_client

    stream_iter = await model.complete(
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


@pytest.mark.asyncio
async def test_streaming_read_timeout_opt_in_bounds_the_stall_window() -> None:
    """stream_read_timeout, when set, is the streaming per-chunk read timeout.

    Mirrors OpenAIModel: a stall detector that resets on every byte (so a live
    stream is never cut off) yet fails a silently dead connection instead of
    hanging the turn forever — independent of the whole-request read_timeout.
    """
    lines: list[str] = []
    lines.extend(_anthropic_sse("message_start", {"type": "message_start"}))
    lines.extend(_anthropic_sse("message_stop", {"type": "message_stop"}))

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(
        return_value=_MockStreamContext(_MockStreamResponse(lines))
    )

    model = ClaudeModel(
        config=_anthropic_config(api_key="test-key"),
        read_timeout=90.0,
        stream_read_timeout=30.0,
    )
    model._client = mock_client

    stream_iter = await model.complete(
        messages=[Message(role="user", content="hello")],
        stream=True,
    )
    _ = [delta async for delta in stream_iter]

    timeout = mock_client.stream.call_args[1]["timeout"]
    # The stall window is the opt-in value, not the 90s whole-request read.
    assert timeout.read == 30.0


# ---------------------------------------------------------------------------
# ISSUE-1: Anthropic prompt-caching breakpoints
# ---------------------------------------------------------------------------


def _caching_model() -> ClaudeModel:
    return ClaudeModel(
        config=_anthropic_config(enable_prompt_caching=True),
        model="claude-3-haiku-20240307",
    )


def test_caching_marks_stable_system_block_not_volatile() -> None:
    model = _caching_model()
    messages = [
        Message(role="system", content="STABLE PREFIX", cache_control=True),
        Message(role="system", content="VOLATILE TAIL"),
        Message(role="user", content="Hello"),
    ]

    system, _ = model._build_messages(messages)

    assert system == [
        {
            "type": "text",
            "text": "STABLE PREFIX",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "VOLATILE TAIL"},
    ]


def test_caching_marks_last_message_content_block() -> None:
    model = _caching_model()
    messages = [
        Message(role="system", content="STABLE", cache_control=True),
        Message(role="user", content="first"),
        Message(role="user", content="latest"),
    ]

    _, anthropic_messages = model._build_messages(messages)

    # Only the last message's last content block carries the breakpoint.
    assert "cache_control" not in anthropic_messages[0]["content"][-1]
    assert anthropic_messages[-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }


def test_caching_marks_only_last_tool() -> None:
    model = _caching_model()
    tools = [
        ToolSchema(name="a", description="tool a", parameters={"type": "object"}),
        ToolSchema(name="b", description="tool b", parameters={"type": "object"}),
    ]

    anthropic_tools = model._build_tools(tools)

    assert "cache_control" not in anthropic_tools[0]
    assert anthropic_tools[-1]["cache_control"] == {"type": "ephemeral"}


def test_caching_disabled_keeps_plain_system_string_and_no_breakpoints() -> None:
    model = ClaudeModel(
        config=_anthropic_config(enable_prompt_caching=False),
        model="claude-3-haiku-20240307",
    )
    messages = [
        Message(role="system", content="STABLE", cache_control=True),
        Message(role="user", content="Hello"),
    ]
    tools = [ToolSchema(name="a", description="d", parameters={"type": "object"})]

    system, anthropic_messages = model._build_messages(messages)
    anthropic_tools = model._build_tools(tools)

    assert system == "STABLE"
    assert "cache_control" not in anthropic_messages[-1]["content"][-1]
    assert "cache_control" not in anthropic_tools[-1]


def test_caching_adds_ladder_breakpoint_for_wide_tool_batches() -> None:
    """A wide parallel tool batch appends more content blocks in one request
    than Anthropic's ~20-block lookback can bridge from the single trailing
    breakpoint. The adapter must place one intermediate "ladder" marker within
    the window so the previous request's tail entry stays reachable."""
    model = _caching_model()
    width = 12
    messages = [
        Message(role="system", content="STABLE", cache_control=True),
        Message(role="user", content="weather for twelve cities"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id=f"c{i}", name="lookup", arguments={"city": str(i)})
                for i in range(width)
            ],
        ),
        *[
            Message(role="tool", content=f"r{i}", tool_call_id=f"c{i}")
            for i in range(width)
        ],
        Message(role="assistant", content="done"),
        Message(role="user", content="next question"),
    ]

    _, anthropic_messages = model._build_messages(messages)

    marked_indexes = [
        index
        for index, message in enumerate(anthropic_messages)
        if any("cache_control" in block for block in message["content"])
    ]
    assert anthropic_messages[-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    assert len(marked_indexes) == 2, marked_indexes
    ladder_index = marked_indexes[0]
    blocks_between = sum(
        len(message["content"])
        for message in anthropic_messages[ladder_index + 1 :]
    )
    # The ladder marker must sit within the documented lookback window of the
    # tail so consecutive wide turns keep a reachable entry chain.
    assert 0 < blocks_between <= 20, blocks_between


def test_caching_keeps_single_message_breakpoint_for_short_history() -> None:
    """Short conversations stay within one lookback window — no ladder marker
    (and no extra cache-write cost)."""
    model = _caching_model()
    messages = [
        Message(role="system", content="STABLE", cache_control=True),
        Message(role="user", content="first"),
        Message(role="assistant", content="answer"),
        Message(role="user", content="latest"),
    ]

    _, anthropic_messages = model._build_messages(messages)

    marked = [
        index
        for index, message in enumerate(anthropic_messages)
        if any("cache_control" in block for block in message["content"])
    ]
    assert marked == [len(anthropic_messages) - 1]


def test_caching_ladder_respects_four_breakpoint_budget() -> None:
    """tools(1) + flagged system entries + message markers must not exceed
    Anthropic's 4-breakpoint budget: with two flagged system entries only the
    tail message marker fits."""
    model = _caching_model()
    width = 12
    messages = [
        Message(role="system", content="STABLE A", cache_control=True),
        Message(role="system", content="STABLE B", cache_control=True),
        Message(role="user", content="weather for twelve cities"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id=f"c{i}", name="lookup", arguments={"city": str(i)})
                for i in range(width)
            ],
        ),
        *[
            Message(role="tool", content=f"r{i}", tool_call_id=f"c{i}")
            for i in range(width)
        ],
        Message(role="user", content="next question"),
    ]

    system, anthropic_messages = model._build_messages(messages)

    flagged_system = sum(
        1 for block in system if isinstance(block, dict) and "cache_control" in block
    )
    message_markers = sum(
        1
        for message in anthropic_messages
        if any("cache_control" in block for block in message["content"])
    )
    assert flagged_system == 2
    assert message_markers == 1  # tail only; no room for the ladder
    # 1 (tools) + 2 (system) + 1 (tail) == 4
    assert 1 + flagged_system + message_markers <= 4


def test_caching_headers_have_no_beta_header() -> None:
    # Anthropic prompt caching is GA; no anthropic-beta header is required.
    model = _caching_model()
    headers = model._messages_adapter.headers()
    assert not any(key.lower() == "anthropic-beta" for key in headers)


def test_caching_build_request_body_emits_system_block_list() -> None:
    model = _caching_model()
    from ecs_agent.providers.anthropic_messages_adapter import AnthropicMessagesRequest

    request = AnthropicMessagesRequest(
        messages=[
            Message(role="system", content="STABLE", cache_control=True),
            Message(role="user", content="Hi"),
        ]
    )
    body = model._messages_adapter.build_request_body(request)

    assert isinstance(body["system"], list)
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
