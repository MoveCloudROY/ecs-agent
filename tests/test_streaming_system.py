import asyncio
import pytest
import time

from collections.abc import AsyncIterator
from typing import Any

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PendingToolCallsComponent,
    StreamingComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import (
    CompletionResult,
    Message,
    StreamContentStartEvent,
    StreamDelta,
    StreamContentDeltaEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamReasoningEndEvent,
    StreamStartEvent,
    ToolCall,
    ToolSchema,
)


class _RecordingLogger:
    def __init__(self) -> None:
        self.info_calls: list[tuple[str, dict[str, object]]] = []
        self.error_calls: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **kwargs: object) -> None:
        self.info_calls.append((event, kwargs))

    def error(self, event: str, **kwargs: object) -> None:
        self.error_calls.append((event, kwargs))


class RecordingStreamingFakeProvider(FakeProvider):
    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[tuple[list[Message], list[ToolSchema] | None, bool]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        self.calls.append((list(messages), tools, stream))
        return await super().complete(
            messages,
            tools=tools,
            stream=stream,
            response_format=response_format,
        )


class ToolCallStreamingFakeProvider(FakeProvider):
    async def _stream_complete(
        self, result: CompletionResult
    ) -> AsyncIterator[StreamDelta]:
        _ = result
        yield StreamDelta(content="Need weather ")
        yield StreamDelta(
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="get_weather",
                    arguments={"_partial": '{"city":"Par'},
                )
            ]
        )
        yield StreamDelta(
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="get_weather",
                    arguments={"_partial": 'is"}'},
                )
            ]
        )
        yield StreamDelta(finish_reason="tool_calls")


class FailingStreamingFakeProvider(FakeProvider):
    async def _stream_complete(
        self, result: CompletionResult
    ) -> AsyncIterator[StreamDelta]:
        _ = result
        yield StreamDelta(content="partial")
        raise RuntimeError("stream broke")


class ReasoningContentStreamingFakeProvider(FakeProvider):
    async def _stream_complete(
        self, result: CompletionResult
    ) -> AsyncIterator[StreamDelta]:
        _ = result
        yield StreamDelta(reasoning_content="thinking")
        yield StreamDelta(content="done")
        yield StreamDelta(finish_reason="stop")


@pytest.mark.asyncio
async def test_streaming_enabled_calls_provider_with_stream_true() -> None:
    world = World()
    provider = RecordingStreamingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Hello"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    await ReasoningSystem().process(world)

    assert provider.calls == [([Message(role="user", content="Hi")], None, True)]


@pytest.mark.asyncio
async def test_streaming_produces_complete_message_from_deltas() -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Hello world"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Say hello")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    await ReasoningSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[-1] == Message(role="assistant", content="Hello world")


@pytest.mark.asyncio
async def test_streaming_events_emitted_in_order() -> None:
    world = World()
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="OK"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Go")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    seen: list[str] = []
    deltas: list[str] = []

    async def on_start(event: StreamStartEvent) -> None:
        assert event.entity_id == entity_id
        seen.append("start")

    async def on_delta(event: StreamContentDeltaEvent) -> None:
        assert event.entity_id == entity_id
        seen.append("delta")
        deltas.append(event.delta)

    async def on_end(event: StreamEndEvent) -> None:
        assert event.entity_id == entity_id
        seen.append("end")

    world.event_bus.subscribe(StreamStartEvent, on_start)
    world.event_bus.subscribe(StreamContentDeltaEvent, on_delta)
    world.event_bus.subscribe(StreamEndEvent, on_end)

    await ReasoningSystem().process(world)

    assert seen[0] == "start"
    assert seen[-1] == "end"
    assert deltas == ["O", "K"]


@pytest.mark.asyncio
async def test_streaming_tool_call_deltas_accumulate_into_pending_tool_calls() -> None:
    world = World()
    provider = ToolCallStreamingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content=""))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="Weather in Paris")]
        ),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    await ReasoningSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    conversation = world.get_component(entity_id, ConversationComponent)
    assert pending is not None
    assert pending.tool_calls == [
        ToolCall(id="call-1", name="get_weather", arguments={"city": "Paris"})
    ]
    assert conversation is not None
    assert conversation.messages[-1].tool_calls == pending.tool_calls


@pytest.mark.asyncio
async def test_streaming_error_preserves_partial_content_and_sets_error_component() -> (
    None
):
    world = World()
    provider = FailingStreamingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    await ReasoningSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    error = world.get_component(entity_id, ErrorComponent)
    assert conversation is not None
    assert conversation.messages[-1] == Message(role="assistant", content="partial")
    assert error is not None
    assert error.system_name == "ReasoningSystem"
    assert "stream broke" in error.error


@pytest.mark.asyncio
async def test_without_streaming_component_uses_non_streaming_path() -> None:
    world = World()
    provider = RecordingStreamingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="non-stream"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )

    await ReasoningSystem().process(world)

    assert provider.calls == [([Message(role="user", content="Hi")], None, False)]


@pytest.mark.asyncio
async def test_streaming_component_disabled_uses_non_streaming_path() -> None:
    world = World()
    provider = RecordingStreamingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="still non-stream")
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=False))

    await ReasoningSystem().process(world)

    assert provider.calls == [([Message(role="user", content="Hi")], None, False)]


@pytest.mark.asyncio
async def test_streaming_logs_first_sse_and_first_content_delta_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ecs_agent.systems import reasoning as reasoning_module

    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Latency check"))
        ]
    )
    logger = _RecordingLogger()
    monkeypatch.setattr(reasoning_module, "logger", logger)

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Measure")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    await ReasoningSystem().process(world)

    events = [event for event, _ in logger.info_calls]
    assert "reasoning_stream_first_sse_event" in events
    assert "reasoning_stream_first_content_delta" in events

    first_sse_payload = next(
        kwargs
        for event, kwargs in logger.info_calls
        if event == "reasoning_stream_first_sse_event"
    )
    assert isinstance(first_sse_payload["time_to_first_sse_event_ms"], float)
    assert isinstance(first_sse_payload["stream_setup_ms"], float)
    assert isinstance(first_sse_payload["total_to_first_sse_event_ms"], float)
    assert first_sse_payload["time_to_first_sse_event_ms"] >= 0.0

    first_content_payload = next(
        kwargs
        for event, kwargs in logger.info_calls
        if event == "reasoning_stream_first_content_delta"
    )
    assert isinstance(first_content_payload["time_to_first_content_delta_ms"], float)
    assert isinstance(first_content_payload["stream_setup_ms"], float)
    assert isinstance(first_content_payload["total_to_first_content_delta_ms"], float)
    assert first_content_payload["time_to_first_content_delta_ms"] >= 0.0


@pytest.mark.asyncio
async def test_streaming_non_blocking_delta_publish_avoids_handler_backpressure() -> (
    None
):
    world = World()
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="ABCD"))]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(
        entity_id,
        StreamingComponent(enabled=True, non_blocking_delta_publish=True),
    )

    seen: list[str] = []

    async def slow_handler(event: StreamContentDeltaEvent) -> None:
        await asyncio.sleep(0.05)
        seen.append(event.delta)

    world.event_bus.subscribe(StreamContentDeltaEvent, slow_handler)

    start = time.perf_counter()
    await ReasoningSystem().process(world)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.12

    await asyncio.sleep(0.25)
    assert seen == ["A", "B", "C", "D"]


@pytest.mark.asyncio
async def test_streaming_publishes_reasoning_and_content_deltas_separately() -> None:
    world = World()
    provider = ReasoningContentStreamingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    reasoning_seen: list[str] = []
    content_seen: list[str] = []

    async def on_reasoning(event: StreamReasoningDeltaEvent) -> None:
        reasoning_seen.append(event.reasoning_delta)

    async def on_content(event: StreamContentDeltaEvent) -> None:
        content_seen.append(event.delta)

    world.event_bus.subscribe(StreamReasoningDeltaEvent, on_reasoning)
    world.event_bus.subscribe(StreamContentDeltaEvent, on_content)

    await ReasoningSystem().process(world)

    assert reasoning_seen == ["thinking"]
    assert content_seen == ["done"]


@pytest.mark.asyncio
async def test_streaming_emits_reasoning_end_then_content_start_transition_events() -> (
    None
):
    world = World()
    provider = ReasoningContentStreamingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    seen: list[str] = []

    async def on_reasoning_delta(event: StreamReasoningDeltaEvent) -> None:
        seen.append(f"reasoning_delta:{event.reasoning_delta}")

    async def on_reasoning_end(event: StreamReasoningEndEvent) -> None:
        _ = event
        seen.append("reasoning_end")

    async def on_content_start(event: StreamContentStartEvent) -> None:
        _ = event
        seen.append("content_start")

    async def on_content_delta(event: StreamContentDeltaEvent) -> None:
        seen.append(f"content_delta:{event.delta}")

    world.event_bus.subscribe(StreamReasoningDeltaEvent, on_reasoning_delta)
    world.event_bus.subscribe(StreamReasoningEndEvent, on_reasoning_end)
    world.event_bus.subscribe(StreamContentStartEvent, on_content_start)
    world.event_bus.subscribe(StreamContentDeltaEvent, on_content_delta)

    await ReasoningSystem().process(world)

    assert seen == [
        "reasoning_delta:thinking",
        "reasoning_end",
        "content_start",
        "content_delta:done",
    ]


@pytest.mark.asyncio
async def test_streaming_contract_emits_single_start_and_end_around_reasoning_and_content() -> (
    None
):
    world = World()
    provider = ReasoningContentStreamingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    seen: list[str] = []

    async def on_start(event: StreamStartEvent) -> None:
        assert event.entity_id == entity_id
        seen.append("start")

    async def on_reasoning(event: StreamReasoningDeltaEvent) -> None:
        assert event.entity_id == entity_id
        seen.append(f"reasoning:{event.reasoning_delta}")

    async def on_content(event: StreamContentDeltaEvent) -> None:
        assert event.entity_id == entity_id
        seen.append(f"content:{event.delta}")

    async def on_end(event: StreamEndEvent) -> None:
        assert event.entity_id == entity_id
        seen.append("end")

    world.event_bus.subscribe(StreamStartEvent, on_start)
    world.event_bus.subscribe(StreamReasoningDeltaEvent, on_reasoning)
    world.event_bus.subscribe(StreamContentDeltaEvent, on_content)
    world.event_bus.subscribe(StreamEndEvent, on_end)

    await ReasoningSystem().process(world)

    assert seen == ["start", "reasoning:thinking", "content:done", "end"]
