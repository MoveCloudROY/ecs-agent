"""Model completion semantics survive streaming, accounting, and persistence."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from ecs_agent.accounting.models import LLMInvocationEvent, StreamCompleteness
from ecs_agent.accounting.instrumentation import complete_with_llm_invocation_event
from ecs_agent.components import (
    ConversationComponent,
    InterruptionComponent,
    LLMComponent,
    PendingToolCallsComponent,
    StreamingComponent,
    TerminalComponent,
)
from ecs_agent.observability.events import LLMObservationCompletedEvent
from ecs_agent.outcomes import GenerationStatus
from ecs_agent.serialization import WorldSerializer
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import FinishReason, ReasoningCompleteEvent, StreamEndEvent
from ecs_agent.types import (
    CompletionResult,
    InterruptionReason,
    Message,
    StreamContentDeltaEvent,
    StreamDelta,
    Usage,
)
from ecs_agent.providers.fake_model import FakeModel
from tests.test_message_flow_contracts import reply, sdk_world


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("status", "detail", "generation", "finish"),
    [
        ("completed", None, GenerationStatus.COMPLETED, FinishReason.STOP),
        ("incomplete", "max_output_tokens", GenerationStatus.INCOMPLETE, FinishReason.LENGTH),
        ("incomplete", "content_filter", GenerationStatus.FILTERED, FinishReason.CONTENT_FILTER),
        ("cancelled", None, GenerationStatus.CANCELLED, FinishReason.CANCELLED),
        ("incomplete", "unrecognized", GenerationStatus.UNKNOWN, FinishReason.UNKNOWN),
        ("failed", None, GenerationStatus.ERROR, FinishReason.ERROR),
    ],
)
async def test_generation_outcome_is_consistent_across_consumers(
    stream: bool,
    status: str,
    detail: str | None,
    generation: GenerationStatus,
    finish: FinishReason,
) -> None:
    world, entity, model, transport = await sdk_world()
    data = reply("resp_outcome", status=status).json()
    if status == "failed":
        data["error"] = {"code": "server_error", "message": "generation failed"}
    if detail is not None:
        data["incomplete_details"] = {"reason": detail}
    if stream:
        world.add_component(entity, StreamingComponent(enabled=True))
        events = [
            {"type": "response.output_text.delta", "delta": "partial answer"},
            {"type": f"response.{status}", "response": data},
        ]
        transport.responses = [httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )]
    else:
        transport.responses = [httpx.Response(200, json=data)]
    recorded: list[object] = []

    async def record(event: object) -> None:
        recorded.append(event)

    for event_type in (
        LLMInvocationEvent, LLMObservationCompletedEvent, StreamEndEvent,
        ReasoningCompleteEvent,
    ):
        world.event_bus.subscribe(event_type, record)
    try:
        await ReasoningSystem().process(world)
        conversation = world.get_component(entity, ConversationComponent)
        assert conversation is not None
        message = conversation.messages[-1]
        assert message.content == "partial answer"
        assert message.generation_status is generation
        assert message.finish_reason is finish
        assert not world.has_component(entity, PendingToolCallsComponent)
        terminal = world.get_component(entity, TerminalComponent)
        assert terminal is not None
        expected_reason = (
            "reasoning_complete" if generation is GenerationStatus.COMPLETED
            else f"reasoning_{generation.value}"
        )
        assert terminal.reason == expected_reason
        assert sum(isinstance(e, ReasoningCompleteEvent) for e in recorded) == (
            1 if generation is GenerationStatus.COMPLETED else 0
        )
        for event in recorded:
            if isinstance(event, (LLMInvocationEvent, LLMObservationCompletedEvent, StreamEndEvent)):
                assert event.generation_status is generation
                assert event.finish_reason == finish
                assert event.status == generation.invocation_status
            if isinstance(event, LLMInvocationEvent):
                # A truncated answer can still have a complete usage report.
                assert event.usage.stream_completeness is StreamCompleteness.COMPLETE
                assert event.usage.prompt_tokens == 2000
        restored = WorldSerializer._message_from_dict(
            WorldSerializer._message_to_dict(message)
        )
        assert restored == message
    finally:
        await model.aclose()


async def test_eof_without_terminal_event_is_unknown_and_usage_is_missing() -> None:
    world, entity, model, transport = await sdk_world()
    world.add_component(entity, StreamingComponent(enabled=True))
    transport.responses = [httpx.Response(
        200, headers={"content-type": "text/event-stream"},
        content='data: {"type":"response.output_text.delta","delta":"unfinished"}\n\n',
    )]
    invocations: list[LLMInvocationEvent] = []

    async def record(event: LLMInvocationEvent) -> None:
        invocations.append(event)

    world.event_bus.subscribe(LLMInvocationEvent, record)
    try:
        await ReasoningSystem().process(world)
        assert len(invocations) == 1
        assert invocations[0].generation_status is GenerationStatus.UNKNOWN
        assert invocations[0].usage.stream_completeness is StreamCompleteness.UNKNOWN
        assert invocations[0].usage.prompt_tokens is None
    finally:
        await model.aclose()


@pytest.mark.parametrize("status", ["incomplete", "cancelled"])
async def test_non_complete_function_call_is_not_enqueued_or_replayed(status: str) -> None:
    world, entity, model, transport = await sdk_world()
    transport.responses = [reply("resp_partial_tool", status=status, output=[{
        "type": "function_call", "call_id": "call_write", "name": "write",
        "arguments": '{"path":"output.txt"}',
    }])]
    try:
        await ReasoningSystem().process(world)
        assert not world.has_component(entity, PendingToolCallsComponent)
        conversation = world.get_component(entity, ConversationComponent)
        assert conversation is not None
        assert conversation.messages[-1].tool_calls is None
        assert conversation.messages[-1].generation_status is not GenerationStatus.COMPLETED
    finally:
        await model.aclose()


@pytest.mark.parametrize("arguments", ['{"path":', '[1,2]', 'null'])
async def test_malformed_arguments_never_become_empty_executable_arguments(arguments: str) -> None:
    world, entity, model, transport = await sdk_world()
    transport.responses = [reply("resp_invalid_tool", output=[{
        "type": "function_call", "call_id": "call_write", "name": "write",
        "arguments": arguments,
    }])]
    try:
        await ReasoningSystem().process(world)
        assert not world.has_component(entity, PendingToolCallsComponent)
        terminal = world.get_component(entity, TerminalComponent)
        assert terminal is not None and terminal.reason == "reasoning_error"
    finally:
        await model.aclose()


async def test_auxiliary_stream_accounting_waits_for_terminal_and_preserves_usage() -> None:
    world, entity, model, transport = await sdk_world()
    data = reply("resp_auxiliary", status="incomplete").json()
    transport.responses = [httpx.Response(
        200, headers={"content-type": "text/event-stream"},
        content="data: " + json.dumps({"type": "response.incomplete", "response": data}) + "\n\n",
    )]
    events: list[LLMInvocationEvent] = []

    async def record(event: LLMInvocationEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(LLMInvocationEvent, record)
    try:
        result = await complete_with_llm_invocation_event(
            event_bus=world.event_bus, entity_id=entity, model=model,
            messages=[], operation="planning", stream=True,
        )
        assert not events
        assert not isinstance(result, CompletionResult)
        _ = [delta async for delta in result]
        assert len(events) == 1
        assert events[0].generation_status is GenerationStatus.INCOMPLETE
        assert events[0].usage.prompt_tokens == 2000
        assert events[0].request_id == "resp_auxiliary"
    finally:
        await model.aclose()


@pytest.mark.parametrize("interrupted", [False, True])
async def test_stream_error_or_interrupt_preserves_already_reported_usage(interrupted: bool) -> None:
    world, entity, sdk, _ = await sdk_world()

    class UsageThenError(FakeModel):
        """Yield provider usage before an interruption or transport failure."""

        async def _stream_complete(self, result: CompletionResult) -> AsyncIterator[StreamDelta]:
            yield StreamDelta(
                usage=Usage(prompt_tokens=100, stream_completeness=StreamCompleteness.PARTIAL),
                response_id="resp_early_usage",
            )
            yield StreamDelta(content="unfinished")
            raise httpx.ReadError("connection closed after partial output")

    llm = world.get_component(entity, LLMComponent)
    assert llm is not None
    llm.model = UsageThenError(responses=[
        CompletionResult(message=Message(role="assistant", content="unused"))
    ])
    world.add_component(entity, StreamingComponent(enabled=True))
    events: list[LLMInvocationEvent] = []

    async def record(event: LLMInvocationEvent) -> None:
        events.append(event)

    async def interrupt(event: StreamContentDeltaEvent) -> None:
        if interrupted:
            world.add_component(entity, InterruptionComponent(reason=InterruptionReason.USER_REQUESTED))

    world.event_bus.subscribe(LLMInvocationEvent, record)
    world.event_bus.subscribe(StreamContentDeltaEvent, interrupt)
    try:
        await ReasoningSystem().process(world)
        assert len(events) == 1
        assert events[0].status == ("cancelled" if interrupted else "error")
        assert events[0].usage.prompt_tokens == 100
        assert events[0].usage.stream_completeness is StreamCompleteness.PARTIAL
        assert events[0].request_id == "resp_early_usage"
    finally:
        await sdk.aclose()
