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

    assert deltas[0].tool_calls is not None
    assert deltas[0].tool_calls[0].id == "call_1"
    assert deltas[0].tool_calls[0].name == "get_weather"
    assert deltas[0].tool_calls[0].arguments == {"_partial": '{"city":"'}

    assert deltas[1].tool_calls is not None
    assert deltas[1].tool_calls[0].arguments == {"city": "NYC"}


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
    assert final_delta.finish_reason == "stop"
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
