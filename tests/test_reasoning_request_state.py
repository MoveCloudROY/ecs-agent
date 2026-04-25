import asyncio
from collections.abc import AsyncIterator

import pytest

from ecs_agent.accounting.models import LLMInvocationEvent, StreamCompleteness
from ecs_agent.components import (
    ConversationComponent,
    InterruptionComponent,
    LLMComponent,
    StreamingComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import FakeModel
from ecs_agent.systems import reasoning as reasoning_module
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message, StreamDelta, Usage


class _RecordingLogger:
    def __init__(self) -> None:
        self.info_calls: list[tuple[str, dict[str, object]]] = []

    def info(self, event_name: str, **kwargs: object) -> None:
        self.info_calls.append((event_name, kwargs))

    def error(self, event_name: str, **kwargs: object) -> None:
        _ = event_name
        _ = kwargs


class _CancelledStreamingFakeModel(FakeModel):
    async def _stream_complete(
        self,
        result: CompletionResult,
    ) -> AsyncIterator[StreamDelta]:
        _ = result
        yield StreamDelta(content="partial")
        raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_non_streaming_emits_single_llm_invocation_event_with_active_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    provider = FakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="done"),
                usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )
        ]
    )
    logger = _RecordingLogger()
    monkeypatch.setattr(reasoning_module, "logger", logger)

    active_model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="done"),
                usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )
        ],
        model_id="active-model",
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    llm_component = world.get_component(entity_id, LLMComponent)
    assert llm_component is not None
    llm_component.pending_model = active_model

    invocation_events: list[LLMInvocationEvent] = []

    async def on_invocation(event: LLMInvocationEvent) -> None:
        invocation_events.append(event)

    world.event_bus.subscribe(LLMInvocationEvent, on_invocation)

    await ReasoningSystem().process(world)

    assert len(invocation_events) == 1
    event = invocation_events[0]
    assert event.model == "active-model"
    assert event.usage.stream_completeness is StreamCompleteness.COMPLETE
    assert event.usage.prompt_tokens == 3
    assert event.usage.completion_tokens == 2
    assert event.usage.total_tokens == 5

    start_payload = next(
        kwargs
        for event_name, kwargs in logger.info_calls
        if event_name == "reasoning_start"
    )
    assert start_payload["model"] == event.model


@pytest.mark.asyncio
async def test_streaming_success_emits_single_complete_llm_invocation_event() -> None:
    world = World()
    provider = FakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="AB"),
                usage=Usage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
            )
        ],
        model_id="stream-model",
    )

    entity_id = world.create_entity()
    world.add_component(
        entity_id, LLMComponent(model=provider)
    )
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    invocation_events: list[LLMInvocationEvent] = []

    async def on_invocation(event: LLMInvocationEvent) -> None:
        invocation_events.append(event)

    world.event_bus.subscribe(LLMInvocationEvent, on_invocation)

    await ReasoningSystem().process(world)

    assert len(invocation_events) == 1
    event = invocation_events[0]
    assert event.model == "stream-model"
    assert event.usage.stream_completeness is StreamCompleteness.COMPLETE
    assert event.cost is None


@pytest.mark.asyncio
async def test_interrupted_stream_emits_single_partial_or_unknown_llm_invocation_event() -> (
    None
):
    world = World()
    provider = _CancelledStreamingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="ignored"),
                usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            )
        ],
        model_id="interrupt-model",
    )

    entity_id = world.create_entity()
    world.add_component(
        entity_id, LLMComponent(model=provider)
    )
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))

    invocation_events: list[LLMInvocationEvent] = []

    async def on_invocation(event: LLMInvocationEvent) -> None:
        invocation_events.append(event)

    world.event_bus.subscribe(LLMInvocationEvent, on_invocation)

    with pytest.raises(asyncio.CancelledError):
        await ReasoningSystem().process(world)

    interruption = world.get_component(entity_id, InterruptionComponent)
    assert interruption is not None
    assert len(invocation_events) == 1
    event = invocation_events[0]
    assert event.model == "interrupt-model"
    assert event.cost is None
    assert event.usage.stream_completeness in {
        StreamCompleteness.PARTIAL,
        StreamCompleteness.UNKNOWN,
    }
    assert event.usage.prompt_tokens is None
    assert event.usage.completion_tokens is None
    assert event.usage.total_tokens is None
