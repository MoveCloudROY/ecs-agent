"""Incomplete auxiliary generations must not mutate valid task state."""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from ecs_agent.accounting.instrumentation import complete_with_llm_invocation_event
from ecs_agent.accounting.models import LLMInvocationEvent, StreamCompleteness
from ecs_agent.components import (
    CompactionConfigComponent, ConversationArchiveComponent, ConversationComponent,
    CurrentCompactionSummaryComponent, LLMComponent, PendingToolCallsComponent,
    PlanComponent, StreamingComponent, TerminalComponent, ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.observability.events import LLMObservationCompletedEvent
from ecs_agent.outcomes import GenerationStatus
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.replanning import ReplanningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, FinishReason, Message, StreamDelta, ToolCall, ToolSchema
from tests.http_mock import sse_response
from tests.test_message_flow_contracts import reply, sdk_world
from examples.e2e.plan_and_task.runtime import derive_workflow_id_from_llm, slug_from_description


@pytest.mark.parametrize("status", ["completed", "incomplete"])
async def test_workflow_id_generation_is_accounted_and_partial_slug_is_not_used(status: str) -> None:
    world, entity, model, transport = await sdk_world()
    transport.responses = [reply("resp_slug", status=status, output=[{
        "type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "generated-slug"},
        ],
    }])]
    events: list[LLMInvocationEvent] = []

    async def record(event: LLMInvocationEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(LLMInvocationEvent, record)
    try:
        slug = await derive_workflow_id_from_llm("tiny task", model, world=world, entity_id=entity)
        assert slug == ("generated-slug" if status == "completed" else slug_from_description("tiny task"))
        assert len(events) == 1 and events[0].operation == "workflow_id"
        assert events[0].request_id == "resp_slug" and events[0].usage.prompt_tokens == 2000
        assert events[0].status == ("success" if status == "completed" else "incomplete")
    finally:
        await model.aclose()


@pytest.mark.parametrize("stop", ["close", "cancel", "unstarted"])
async def test_auxiliary_stream_closes_http_and_reports_one_terminal(stop: str) -> None:
    world, entity, model, transport = await sdk_world()
    waiting = asyncio.Event()

    async def lines() -> AsyncIterator[str]:
        yield 'data: {"type":"response.output_text.delta","delta":"partial"}'
        waiting.set()
        await asyncio.Event().wait()

    response = sse_response(lines())
    transport.responses = [response]
    events: list[LLMInvocationEvent] = []
    observations: list[LLMObservationCompletedEvent] = []

    async def record(event: LLMInvocationEvent) -> None:
        events.append(event)

    async def observe(event: LLMObservationCompletedEvent) -> None:
        observations.append(event)

    world.event_bus.subscribe(LLMInvocationEvent, record)
    world.event_bus.subscribe(LLMObservationCompletedEvent, observe)
    try:
        stream = await complete_with_llm_invocation_event(
            event_bus=world.event_bus, entity_id=entity, model=model,
            messages=[], operation="auxiliary", stream=True,
        )
        assert not isinstance(stream, CompletionResult)
        if stop != "unstarted":
            assert (await anext(stream)).content == "partial"
            assert not response.is_closed
        if stop == "cancel":
            task = asyncio.create_task(anext(stream))
            await asyncio.wait_for(waiting.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await stream.aclose()
        await stream.aclose()
        assert len(events) == len(observations) == 1
        assert events[0].generation_status is GenerationStatus.CANCELLED
        assert events[0].status == observations[0].status == "cancelled"
        assert events[0].usage.stream_completeness is StreamCompleteness.UNKNOWN
        assert observations[0].response_message.content == ("" if stop == "unstarted" else "partial")
        if stop != "unstarted":
            assert response.is_closed
        else:
            assert not transport.requests
    finally:
        await model.aclose()


@pytest.mark.parametrize("valid_id", [False, True])
async def test_auxiliary_tool_stream_aggregates_fragments_before_validation(valid_id: bool) -> None:
    world = World()
    entity = world.create_entity()

    class FragmentModel(FakeModel):
        """Emit JSON arguments over multiple deltas without repeating call ID."""

        async def _stream_complete(self, result: CompletionResult) -> AsyncIterator[StreamDelta]:
            yield StreamDelta(tool_calls=[ToolCall(
                id="call_a" if valid_id else "", name="lookup", arguments={"_partial": '{"q":'},
            )])
            yield StreamDelta(tool_calls=[ToolCall(id="", name="", arguments={"_partial": '"value"}'})])
            yield StreamDelta(finish_reason=FinishReason.TOOL_CALLS)

    events: list[LLMObservationCompletedEvent] = []

    async def record(event: LLMObservationCompletedEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(LLMObservationCompletedEvent, record)
    stream = await complete_with_llm_invocation_event(
        event_bus=world.event_bus, entity_id=entity,
        model=FragmentModel([CompletionResult(message=Message(role="assistant", content=""))]),
        messages=[], operation="auxiliary", stream=True,
        tools=[ToolSchema(name="lookup", description="lookup", parameters={
            "type": "object", "required": ["q"], "properties": {"q": {"type": "string"}},
        })],
    )
    assert not isinstance(stream, CompletionResult)
    _ = [delta async for delta in stream]
    assert len(events) == 1
    assert events[0].generation_status is (GenerationStatus.TOOL_CALLS if valid_id else GenerationStatus.ERROR)
    assert events[0].response_message.tool_calls == [ToolCall(
        id="call_a" if valid_id else "", name="lookup", arguments={"q": "value"},
    )]


@pytest.mark.parametrize("finish", [
    FinishReason.LENGTH, FinishReason.CANCELLED, FinishReason.UNKNOWN,
    FinishReason.CONTENT_FILTER, FinishReason.ERROR,
])
@pytest.mark.parametrize("operation", ["planning", "replanning", "compaction"])
async def test_auxiliary_incomplete_generation_cannot_commit_success(
    finish: FinishReason, operation: str,
) -> None:
    world = World()
    entity = world.create_entity()
    messages = [Message(role="user", content="important task " * 100),
                Message(role="assistant", content="existing information " * 100)]
    conversation = ConversationComponent(messages=list(messages))
    plan = PlanComponent(steps=["done", "original next step"], current_step=1)
    world.add_component(entity, conversation)
    world.add_component(entity, plan)
    world.add_component(entity, CurrentCompactionSummaryComponent(summary="valid summary"))
    world.add_component(entity, CompactionConfigComponent(threshold_tokens=1))
    world.add_component(entity, LLMComponent(model=FakeModel([CompletionResult(
        message=Message(role="assistant", content='{"revised_steps":["unsafe replacement"]}'),
        finish_reason=finish,
    )])))
    systems = {"planning": PlanningSystem(), "replanning": ReplanningSystem(),
               "compaction": CompactionSystem()}
    await systems[operation].process(world)
    assert plan.steps == ["done", "original next step"]
    assert plan.current_step == 1 and not plan.completed
    assert not world.has_component(entity, PendingToolCallsComponent)
    assert world.get_component(entity, CurrentCompactionSummaryComponent).summary == "valid summary"
    assert not world.has_component(entity, ConversationArchiveComponent)
    if operation == "planning":
        assert conversation.messages[-1].generation_status is not GenerationStatus.COMPLETED
    else:
        assert conversation.messages == messages
    if operation != "replanning":
        assert world.has_component(entity, TerminalComponent)


SCHEMA = ToolSchema(name="write", description="write result", parameters={
    "type": "object", "required": ["items"], "additionalProperties": False,
    "$defs": {"item": {"type": "object", "required": ["count"],
                       "properties": {"count": {"type": "integer", "minimum": 1}}}},
    "properties": {"items": {"type": "array", "minItems": 1,
                              "items": {"$ref": "#/$defs/item"}}},
})


@pytest.mark.parametrize("arguments", [
    {}, {"items": []}, {"items": [{"count": "1"}]},
    {"items": [{"count": 0}]}, {"items": [{"count": True}]},
    {"items": [{"count": 1}], "extra": 2},
    {"items": [{"count": 1}]},
])
@pytest.mark.parametrize("stream", [False, True])
async def test_tool_schema_checked_before_queue_and_execution(arguments: dict, stream: bool) -> None:
    world, entity, model, transport = await sdk_world()
    executed: list[dict] = []

    async def write(**kwargs: object) -> str:
        executed.append(kwargs)
        return "done"

    world.add_component(entity, ToolRegistryComponent(tools={"write": SCHEMA}, handlers={"write": write}))
    world.add_component(entity, StreamingComponent(enabled=stream))
    payload = reply("resp_schema", output=[{
        "type": "function_call", "call_id": "call_write", "name": "write",
        "arguments": json.dumps(arguments), "status": "completed",
    }])
    if stream:
        async def lines() -> AsyncIterator[str]:
            yield "data: " + json.dumps({"type": "response.completed", "response": payload.json()})
        transport.responses = [sse_response(lines())]
    else:
        transport.responses = [payload]
    try:
        await ReasoningSystem().process(world)
        valid = arguments == {"items": [{"count": 1}]} and type(arguments["items"][0]["count"]) is int
        assert world.has_component(entity, PendingToolCallsComponent) == valid
        await ToolExecutionSystem().process(world)
        assert executed == ([arguments] if valid else [])
        if not valid:
            assert world.get_component(entity, TerminalComponent).reason == "reasoning_error"
        # The executor independently protects imported/manual queues.
        executed.clear()
        world.add_component(entity, PendingToolCallsComponent([ToolCall(
            id="manual", name="write", arguments=arguments,
        )]))
        await ToolExecutionSystem().process(world)
        assert executed == ([arguments] if valid else [])
    finally:
        await model.aclose()
