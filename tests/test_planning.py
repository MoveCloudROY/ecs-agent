from __future__ import annotations

import pytest

from ecs_agent.components import (
    ContextEntry,
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PlanComponent,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    UserPromptConfigComponent,
    SystemPromptComponent,
    TerminalComponent,
)
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.core import World
from ecs_agent.providers import FakeModel
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.types import CompletionResult, Message, PlanStepCompletedEvent, ToolCall


class RecordingFakeModel(FakeModel):
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


class FlakyRecordingProvider(FakeModel):
    def __init__(self) -> None:
        super().__init__(
            responses=[
                CompletionResult(message=Message(role="assistant", content="ok")),
            ]
        )
        self.calls: list[list[Message]] = []
        self._attempt = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
    ) -> CompletionResult:
        _ = tools
        self.calls.append(list(messages))
        self._attempt += 1
        if self._attempt == 1:
            raise RuntimeError("provider exploded")
        return await super().complete(messages, tools=None)


@pytest.mark.asyncio
async def test_process_advances_one_step_appends_response_and_publishes_event() -> None:
    world = World()
    model = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="step done"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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
    model = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="unused"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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
    model = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="unused"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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

    assert len(model.calls) == 1
    sent = model.calls[0]
    assert sent[0] == Message(
        role="system", content="You are concise", cache_control=True
    )
    assert sent[1] == Message(role="system", content="Step 1/1: inspect state")
    assert sent[2] == Message(role="user", content="hello")


@pytest.mark.asyncio
async def test_tool_calls_attach_pending_tool_calls_component() -> None:
    world = World()
    tool_call = ToolCall(id="call-1", name="lookup", arguments={"q": "x"})
    model = FakeModel(
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
    world.add_component(entity_id, LLMComponent(model=model))
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
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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


@pytest.mark.asyncio
async def test_provider_exhaustion_adds_terminal_component() -> None:
    world = World()
    model = FakeModel(responses=[])
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="start")]),
    )
    world.add_component(entity_id, PlanComponent(steps=["step one"], current_step=0))

    await PlanningSystem().process(world)

    error = world.get_component(entity_id, ErrorComponent)
    terminal = world.get_component(entity_id, TerminalComponent)
    # FakeModel raises IndexError when empty -> caught as model_exhausted
    assert terminal is not None
    assert terminal.reason == "provider_exhausted"
    assert error is None


@pytest.mark.asyncio
async def test_generic_exception_adds_error_component() -> None:
    class ExplodingProvider:
        async def complete(
            self,
            messages: list[Message],
            tools: list[object] | None = None,
        ) -> CompletionResult:
            raise RuntimeError("connection failed")

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id, LLMComponent(model=ExplodingProvider())
    )  # type: ignore[arg-type]
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="start")]),
    )
    world.add_component(entity_id, PlanComponent(steps=["step one"], current_step=0))

    await PlanningSystem().process(world)

    error = world.get_component(entity_id, ErrorComponent)
    assert error is not None
    assert "connection failed" in error.error
    assert error.system_name == "PlanningSystem"
    terminal = world.get_component(entity_id, TerminalComponent)
    assert terminal is not None
    assert terminal.reason == "planning_error"


@pytest.mark.asyncio
async def test_prompt_context_injection_is_transient_for_planning_provider_call() -> (
    None
):
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(entity_id, SystemPromptComponent(content="You are concise"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Plan this")]),
    )
    world.add_component(
        entity_id, PlanComponent(steps=["inspect state"], current_step=0)
    )
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="tool-search-0",
                    priority=30,
                    registration_order=0,
                    source_label="tool:search",
                    content="source: tool\nresult: evidence",
                )
            ]
        ),
    )

    await PlanningSystem().process(world)

    sent = model.calls[0]
    assert sent[0] == Message(
        role="system", content="You are concise", cache_control=True
    )
    assert sent[1] == Message(role="system", content="Step 1/1: inspect state")
    assert sent[2].role == "user"
    assert sent[2].content.endswith("Plan this")
    assert "[PROMPT_CONTEXT_POOL]" not in sent[2].content
    # Pool entries ride a trailing user message (cache-prefix safe).
    assert sent[3].role == "user"
    assert sent[3].content.startswith("[PROMPT_CONTEXT_POOL]")

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content == "Plan this"


@pytest.mark.asyncio
async def test_planning_retry_reuses_reserved_context_then_commits_on_success() -> None:
    world = World()
    model = FlakyRecordingProvider()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Plan this")]),
    )
    world.add_component(
        entity_id, PlanComponent(steps=["inspect state"], current_step=0)
    )
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="tool-search-0",
                    priority=30,
                    registration_order=0,
                    source_label="tool:search",
                    content="source: tool\nresult: evidence",
                )
            ]
        ),
    )

    await PlanningSystem().process(world)

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    reservation = world.get_component(entity_id, PromptContextReservationComponent)
    assert queue is not None
    assert reservation is not None
    assert reservation.reserved_entries != []
    reserved_ids = {entry.entry_id for entry in reservation.reserved_entries}

    queue.entries.append(
        ContextEntry(
            entry_id="subagent-writer-1",
            priority=20,
            registration_order=1,
            source_label="subagent:writer",
            content="source: subagent\nresult: draft",
        )
    )

    await PlanningSystem().process(world)

    first_user = model.calls[0][-1].content
    second_user = model.calls[1][-1].content
    assert first_user == second_user
    assert "source: subagent" not in second_user

    assert world.get_component(entity_id, PromptContextReservationComponent) is None
    remaining_ids = {entry.entry_id for entry in queue.entries}
    assert reserved_ids.isdisjoint(remaining_ids)
    assert "subagent-writer-1" in remaining_ids


@pytest.mark.asyncio
async def test_event_trigger_injection_is_transient_for_planning_provider_call() -> (
    None
):
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(entity_id, SystemPromptComponent(content="You are concise"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Plan this")]),
    )
    world.add_component(
        entity_id, PlanComponent(steps=["inspect state"], current_step=0)
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[TriggerSpec(pattern="Plan", match_mode="keyword", action="inject", content="Prefer successful tool context", priority=0)],
            enable_context_pool=True,
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="tool-search-0",
                    priority=30,
                    registration_order=0,
                    source_label="tool:search",
                    content="source: tool:search\nstatus: success\nresult: evidence\nerror: ",
                )
            ]
        ),
    )

    await PlanningSystem().process(world)

    sent = model.calls[0]
    assert sent[2].role == "user"
    assert sent[2].content.startswith(
        "[PROMPT_INJECT:Plan]\nPrefer successful tool context"
    )
    assert sent[2].content.endswith("Plan this")

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content == "Plan this"
