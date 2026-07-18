import json
import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    InterruptionComponent,
    LLMComponent,
    StreamingComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import FakeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import (
    CompletionResult,
    InterruptionReason,
    Message,
    StreamDelta,
    StreamContentDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
)


def _openai_config(
    *,
    api_key: str = "test-key",
    base_url: str = "https://api.openai.com/v1",
    api_format: ApiFormat = ApiFormat.OPENAI_CHAT_COMPLETIONS,
) -> ProviderConfig:
    return ProviderConfig(
        provider_id="openai",
        base_url=base_url,
        api_key=api_key,
        api_format=api_format,
    )


class _MockStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.is_error = False
        self.raise_for_status = Mock()

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _MockStreamContext:
    def __init__(self, response: _MockStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _MockStreamResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload)}"


@pytest.mark.asyncio
async def test_non_streaming_backward_compatibility() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "plain"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response
    mock_client.stream = Mock()

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    result = await model.complete(
        [Message(role="user", content="hello")], stream=False
    )

    assert isinstance(result, CompletionResult)
    assert result.message.content == "plain"
    assert mock_client.post.called
    assert not mock_client.stream.called


@pytest.mark.asyncio
async def test_streaming_returns_stream_delta_objects() -> None:
    stream_lines = [
        _sse_data({"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]}),
        _sse_data({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="hello")], stream=True
    )
    deltas = [delta async for delta in stream_iter]

    assert len(deltas) == 2
    assert all(isinstance(delta, StreamDelta) for delta in deltas)
    assert deltas[0].content == "Hel"
    assert deltas[1].content == "lo"
    assert deltas[1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_streaming_tool_call_arguments_spanning_chunks_yield_once_parsed() -> None:
    """Tool-call arguments split across SSE chunks surface exactly once, fully parsed.

    Regression: the adapter used to re-emit the full accumulated prefix on every
    chunk as ``{"_partial": ...}``, which downstream merging concatenated into
    garbage that shadowed the correctly parsed arguments.
    """
    stream_lines = [
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"city": "Par'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'is"}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_data({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="weather?")], stream=True
    )
    deltas = [delta async for delta in stream_iter]

    tool_deltas = [delta for delta in deltas if delta.tool_calls is not None]
    assert len(tool_deltas) == 1
    (tool_call,) = tool_deltas[0].tool_calls
    assert tool_call.id == "call_1"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"city": "Paris"}
    assert tool_deltas[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_streaming_tool_calls_flush_at_stream_end_without_finish_reason() -> None:
    """Gateways that omit the finish_reason chunk still get their tool calls."""
    stream_lines = [
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q": ',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="q")], stream=True
    )
    deltas = [delta async for delta in stream_iter]

    tool_deltas = [delta for delta in deltas if delta.tool_calls is not None]
    assert len(tool_deltas) == 1
    (tool_call,) = tool_deltas[0].tool_calls
    assert tool_call.arguments == {"q": "x"}


@pytest.mark.asyncio
async def test_streaming_tool_call_without_arguments_parses_to_empty_dict() -> None:
    """No-argument tool calls yield ``{}``, not a ``{"_partial": ""}`` placeholder."""
    stream_lines = [
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "ping", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_data({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete([Message(role="user", content="p")], stream=True)
    deltas = [delta async for delta in stream_iter]

    tool_deltas = [delta for delta in deltas if delta.tool_calls is not None]
    assert len(tool_deltas) == 1
    (tool_call,) = tool_deltas[0].tool_calls
    assert tool_call.name == "ping"
    assert tool_call.arguments == {}


@pytest.mark.asyncio
async def test_streaming_request_includes_usage_stream_options() -> None:
    """Streaming requests opt into the final usage chunk (OpenAI spec)."""
    stream_lines = [
        _sse_data({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="hello")], stream=True
    )
    _ = [delta async for delta in stream_iter]

    body = mock_client.stream.call_args[1]["json"]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_streaming_usage_chunk_with_empty_choices_yields_usage() -> None:
    """The terminal usage chunk has "choices": [] — it must not crash the
    stream, and its usage (incl. cached tokens) must reach the caller."""
    stream_lines = [
        _sse_data({"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}),
        _sse_data({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse_data(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 8077,
                    "completion_tokens": 5,
                    "total_tokens": 8082,
                    "prompt_tokens_details": {"cached_tokens": 7936},
                },
            }
        ),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="hello")], stream=True
    )
    deltas = [delta async for delta in stream_iter]

    assert [delta.content for delta in deltas if delta.content] == ["Hi"]
    usage_deltas = [delta for delta in deltas if delta.usage is not None]
    assert len(usage_deltas) == 1
    usage = usage_deltas[0].usage
    assert usage is not None
    assert usage.prompt_tokens == 8077
    assert usage.cached_input_tokens == 7936


@pytest.mark.asyncio
async def test_streaming_http_error_surfaces_status_error() -> None:
    """A non-2xx streaming response raises HTTPStatusError with a readable
    body (not ResponseNotRead from the error logger)."""

    class _AsyncBody(httpx.AsyncByteStream):
        def __init__(self, body: bytes) -> None:
            self._body = body

        async def __aiter__(self):
            yield self._body

    error_response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        stream=_AsyncBody(b'{"error": {"message": "bad request"}}'),
    )

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(error_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="hello")], stream=True
    )
    with pytest.raises(httpx.HTTPStatusError):
        async for _delta in stream_iter:
            pass


@pytest.mark.asyncio
async def test_streaming_sse_content_chunks() -> None:
    stream_lines = [
        "",
        _sse_data(
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        ),
        _sse_data(
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]}
        ),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="hello")], stream=True
    )
    deltas = [delta async for delta in stream_iter]

    assert [delta.content for delta in deltas] == ["Hello", " world"]


@pytest.mark.asyncio
async def test_streaming_preserves_reasoning_content_separately_from_content() -> None:
    stream_lines = [
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "content": None,
                            "reasoning_content": "Thinking",
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "Answer",
                            "reasoning_content": None,
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="hello")], stream=True
    )
    deltas = [delta async for delta in stream_iter]

    assert len(deltas) == 2
    assert deltas[0].reasoning_content == "Thinking"
    assert deltas[0].content is None
    assert deltas[1].content == "Answer"
    assert deltas[1].reasoning_content is None


@pytest.mark.asyncio
async def test_streaming_accumulates_tool_call_chunks_by_index() -> None:
    stream_lines = [
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'NYC"}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_data({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="weather")], stream=True
    )
    deltas = [delta async for delta in stream_iter]

    tool_deltas = [delta for delta in deltas if delta.tool_calls is not None]
    assert len(tool_deltas) == 1
    (tool_call,) = tool_deltas[0].tool_calls
    assert tool_call.id == "call_1"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"city": "NYC"}


@pytest.mark.asyncio
async def test_done_sentinel_stops_iteration() -> None:
    stream_lines = [
        _sse_data(
            {"choices": [{"delta": {"content": "first"}, "finish_reason": None}]}
        ),
        "data: [DONE]",
        _sse_data(
            {"choices": [{"delta": {"content": "ignored"}, "finish_reason": None}]}
        ),
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(config=_openai_config(api_key="test-key"))
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="x")], stream=True
    )
    deltas = [delta async for delta in stream_iter]

    assert len(deltas) == 1
    assert deltas[0].content == "first"


@pytest.mark.asyncio
async def test_streaming_timeout_configuration() -> None:
    stream_lines = [
        _sse_data(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", base_url="https://test.openai.com/v1")
    )
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="x")], stream=True
    )
    _ = [delta async for delta in stream_iter]

    stream_call = mock_client.stream.call_args
    assert stream_call[0][0] == "POST"
    assert stream_call[0][1] == "https://test.openai.com/v1/chat/completions"
    assert stream_call[1]["json"]["stream"] is True

    timeout = stream_call[1]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 10.0
    assert timeout.read is None
    assert timeout.write == 10.0
    assert timeout.pool == 10.0


@pytest.mark.asyncio
async def test_streaming_timeout_uses_provider_custom_timeout() -> None:
    stream_lines = [
        _sse_data({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]
    stream_response = _MockStreamResponse(stream_lines)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(return_value=_MockStreamContext(stream_response))

    model = OpenAIModel(
        config=_openai_config(api_key="test-key"),
        connect_timeout=4.0,
        read_timeout=90.0,
        write_timeout=7.0,
        pool_timeout=3.0,
    )
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="x")], stream=True
    )
    _ = [delta async for delta in stream_iter]

    timeout = mock_client.stream.call_args[1]["timeout"]
    assert timeout.connect == 4.0
    assert timeout.read is None
    assert timeout.write == 7.0
    assert timeout.pool == 3.0


@pytest.mark.asyncio
async def test_streaming_read_timeout_opt_in_bounds_the_stall_window() -> None:
    """stream_read_timeout, when set, becomes the per-chunk read timeout.

    It is a stall detector: httpx resets it on every streamed byte, so a live
    stream is never cut off, but a silently stalled connection fails instead of
    hanging forever. It stays independent of the whole-request read_timeout so
    bounding request time never caps streaming reasoning.
    """
    stream_lines = [
        _sse_data({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = Mock(
        return_value=_MockStreamContext(_MockStreamResponse(stream_lines))
    )

    model = OpenAIModel(
        config=_openai_config(api_key="test-key"),
        read_timeout=90.0,
        stream_read_timeout=30.0,
    )
    model._client = mock_client

    stream_iter = await model.complete(
        [Message(role="user", content="x")], stream=True
    )
    _ = [delta async for delta in stream_iter]

    timeout = mock_client.stream.call_args[1]["timeout"]
    # The stall window is the opt-in value, not the 90s whole-request read.
    assert timeout.read == 30.0


# FakeModel Streaming Tests


@pytest.mark.asyncio
async def test_fake_model_streams_response_as_character_deltas() -> None:
    """FakeModel should stream response character-by-character when stream=True."""
    from ecs_agent.providers import FakeModel
    from ecs_agent.types import Usage

    msg = Message(role="assistant", content="Hi!")
    usage = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    result = CompletionResult(message=msg, usage=usage)

    model = FakeModel(responses=[result])
    stream_iter = await model.complete(
        [Message(role="user", content="hello")], stream=True
    )

    deltas = [delta async for delta in stream_iter]

    # Should have one delta per character plus final delta with finish_reason
    assert len(deltas) == 4  # 'H', 'i', '!' (3 chars) + final chunk with finish_reason
    assert [d.content for d in deltas[:3]] == ["H", "i", "!"]
    assert deltas[3].content is None
    assert deltas[3].finish_reason == "stop"
    assert deltas[3].usage is not None
    assert deltas[3].usage.total_tokens == 3


@pytest.mark.asyncio
async def test_fake_model_streaming_with_tool_calls() -> None:
    """FakeModel streaming should preserve tool calls in final message."""
    from ecs_agent.providers import FakeModel
    from ecs_agent.types import ToolCall, Usage

    tool_call = ToolCall(id="tc1", name="search", arguments={"q": "test"})
    msg = Message(role="assistant", content="Found", tool_calls=[tool_call])
    usage = Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    result = CompletionResult(message=msg, usage=usage)

    model = FakeModel(responses=[result])
    stream_iter = await model.complete(
        [Message(role="user", content="search")], stream=True
    )

    deltas = [delta async for delta in stream_iter]

    # First 5 deltas are characters 'F', 'o', 'u', 'n', 'd'
    assert len(deltas) == 6
    char_deltas = deltas[:5]
    final_delta = deltas[5]

    assert [d.content for d in char_deltas] == ["F", "o", "u", "n", "d"]
    assert final_delta.tool_calls == [tool_call]
    assert final_delta.finish_reason == "tool_calls"
    assert final_delta.usage == usage


@pytest.mark.asyncio
async def test_fake_model_streaming_empty_content() -> None:
    """FakeModel should handle empty content gracefully."""
    from ecs_agent.providers import FakeModel
    from ecs_agent.types import Usage

    msg = Message(role="assistant", content="")
    usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    result = CompletionResult(message=msg, usage=usage)

    model = FakeModel(responses=[result])
    stream_iter = await model.complete(
        [Message(role="user", content="hi")], stream=True
    )

    deltas = [delta async for delta in stream_iter]

    # Empty content should still yield final chunk with usage
    assert len(deltas) == 1
    assert deltas[0].finish_reason == "stop"
    assert deltas[0].usage == usage


class _CancelledStreamingFakeModel(FakeModel):
    async def _stream_complete(self, result: CompletionResult):
        _ = result
        yield StreamDelta(content="partial")
        raise asyncio.CancelledError()


class _ChunkedStreamingFakeModel(FakeModel):
    def __init__(self, responses: list[CompletionResult], chunks: list[str]) -> None:
        super().__init__(responses=responses)
        self._chunks = chunks

    async def _stream_complete(self, result: CompletionResult):
        _ = result
        for chunk in self._chunks:
            yield StreamDelta(content=chunk)
        yield StreamDelta(finish_reason="stop")


@pytest.mark.asyncio
async def test_streaming_partial_content_persisted_on_interrupt_cancel() -> None:
    world = World()
    model = _CancelledStreamingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    with pytest.raises(asyncio.CancelledError):
        await ReasoningSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    interruption = world.get_component(entity_id, InterruptionComponent)
    error = world.get_component(entity_id, ErrorComponent)

    assert conversation is not None
    assert conversation.messages[-1] == Message(role="assistant", content="partial")
    assert interruption is not None
    assert interruption.reason == InterruptionReason.USER_REQUESTED
    assert error is None


@pytest.mark.asyncio
async def test_streaming_partial_reraises_cancelled_after_interrupt_cleanup() -> None:
    world = World()
    model = _CancelledStreamingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    with pytest.raises(asyncio.CancelledError):
        await ReasoningSystem().process(world)

    interruption = world.get_component(entity_id, InterruptionComponent)
    assert interruption is not None


@pytest.mark.asyncio
async def test_streaming_interrupt_mid_generation_preserves_partial_and_emits_stream_end() -> (
    None
):
    world = World()
    model = _ChunkedStreamingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ],
        chunks=["A", "B", "C"],
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    seen_events: list[str] = []

    async def on_start(event: StreamStartEvent) -> None:
        _ = event
        seen_events.append("start")

    async def on_delta(event: StreamContentDeltaEvent) -> None:
        seen_events.append(f"delta:{event.delta}")
        if event.entity_id == entity_id and event.delta == "A":
            world.add_component(
                entity_id,
                InterruptionComponent(
                    reason=InterruptionReason.SYSTEM_PAUSE,
                    message="pause requested",
                ),
            )

    async def on_end(event: StreamEndEvent) -> None:
        _ = event
        seen_events.append("end")

    world.event_bus.subscribe(StreamStartEvent, on_start)
    world.event_bus.subscribe(StreamContentDeltaEvent, on_delta)
    world.event_bus.subscribe(StreamEndEvent, on_end)

    with pytest.raises(asyncio.CancelledError):
        await ReasoningSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    interruption = world.get_component(entity_id, InterruptionComponent)
    error = world.get_component(entity_id, ErrorComponent)

    assert conversation is not None
    assert conversation.messages[-1] == Message(role="assistant", content="A")
    assert interruption is not None
    assert interruption.reason == InterruptionReason.SYSTEM_PAUSE
    assert error is None
    assert seen_events == ["start", "delta:A", "end"]


@pytest.mark.asyncio
async def test_streaming_interrupt_component_preexisting_skips_streaming_safely() -> (
    None
):
    world = World()
    model = _ChunkedStreamingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ],
        chunks=["A", "B"],
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.add_component(
        entity_id,
        InterruptionComponent(
            reason=InterruptionReason.SYSTEM_PAUSE, message="already idle"
        ),
    )

    seen_events: list[str] = []

    async def on_start(event: StreamStartEvent) -> None:
        _ = event
        seen_events.append("start")

    world.event_bus.subscribe(StreamStartEvent, on_start)

    await ReasoningSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == [Message(role="user", content="hello")]
    assert seen_events == []
