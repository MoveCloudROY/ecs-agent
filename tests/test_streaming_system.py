import pytest

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
    StreamDelta,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    ToolCall,
    ToolSchema,
)


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

    async def on_delta(event: StreamDeltaEvent) -> None:
        assert event.entity_id == entity_id
        seen.append("delta")
        deltas.append(event.delta)

    async def on_end(event: StreamEndEvent) -> None:
        assert event.entity_id == entity_id
        seen.append("end")

    world.event_bus.subscribe(StreamStartEvent, on_start)
    world.event_bus.subscribe(StreamDeltaEvent, on_delta)
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
