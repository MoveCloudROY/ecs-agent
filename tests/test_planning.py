from __future__ import annotations

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PlanComponent,
    SystemPromptComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.types import CompletionResult, Message, PlanStepCompletedEvent, ToolCall


class RecordingFakeProvider(FakeProvider):
    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
    ) -> CompletionResult:
        _ = tools
        self.calls.append(list(messages))
        return await super().complete(messages, tools=None)


@pytest.mark.asyncio
async def test_process_advances_one_step_appends_response_and_publishes_event() -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="step done"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="start")]),
    )
    world.add_component(
        entity_id,
        PlanComponent(steps=["gather facts", "answer user"], current_step=0),
    )

    seen: list[PlanStepCompletedEvent] = []

    async def handler(event: PlanStepCompletedEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(PlanStepCompletedEvent, handler)

    await PlanningSystem().process(world)

    plan = world.get_component(entity_id, PlanComponent)
    conversation = world.get_component(entity_id, ConversationComponent)
    assert plan is not None
    assert conversation is not None
    assert plan.current_step == 1
    assert plan.completed is False
    assert conversation.messages[-1].content == "step done"
    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
    assert seen[0].step_index == 0
    assert seen[0].step_description == "gather facts"


@pytest.mark.asyncio
async def test_skips_entity_when_plan_is_completed() -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="unused"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="start")]),
    )
    world.add_component(
        entity_id,
        PlanComponent(steps=["only"], current_step=0, completed=True),
    )

    await PlanningSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert len(conversation.messages) == 1


@pytest.mark.asyncio
async def test_skips_entity_when_plan_steps_are_empty() -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="unused"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="start")]),
    )
    world.add_component(entity_id, PlanComponent(steps=[]))

    await PlanningSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert len(conversation.messages) == 1


@pytest.mark.asyncio
async def test_plan_context_is_injected_before_llm_call() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(entity_id, SystemPromptComponent(content="You are concise"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(
        entity_id,
        PlanComponent(steps=["inspect state"], current_step=0),
    )

    await PlanningSystem().process(world)

    assert len(provider.calls) == 1
    sent = provider.calls[0]
    assert sent[0] == Message(role="system", content="You are concise")
    assert sent[1] == Message(role="system", content="Step 1/1: inspect state")
    assert sent[2] == Message(role="user", content="hello")


@pytest.mark.asyncio
async def test_tool_calls_attach_pending_tool_calls_component() -> None:
    world = World()
    tool_call = ToolCall(id="call-1", name="lookup", arguments='{"q":"x"}')
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[tool_call],
                )
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="use tool")]),
    )
    world.add_component(entity_id, PlanComponent(steps=["use tool"], current_step=0))

    await PlanningSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    assert pending is not None
    assert pending.tool_calls == [tool_call]


@pytest.mark.asyncio
async def test_marks_plan_completed_after_final_step() -> None:
    world = World()
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="finish")]),
    )
    world.add_component(
        entity_id,
        PlanComponent(steps=["final step"], current_step=0),
    )

    await PlanningSystem().process(world)

    plan = world.get_component(entity_id, PlanComponent)
    assert plan is not None
    assert plan.current_step == 1
    assert plan.completed is True
