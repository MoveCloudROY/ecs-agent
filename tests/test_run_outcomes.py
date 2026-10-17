"""Runner and delegation success are derived from the target's terminal state."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from ecs_agent.components import (
    ConversationComponent, ContextTrimConfig, ErrorComponent, LLMComponent, PendingToolCallsComponent, PlanComponent,
    SubagentNotificationQueueComponent, SubagentRegistryComponent,
    SubagentSessionTableComponent, TerminalComponent, ToolRegistryComponent,
)
from ecs_agent.core import Runner, RunStatus, World
from ecs_agent.outcomes import GenerationStatus
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.replanning import ReplanningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent.results import DelegationResult
from ecs_agent.systems.subagent_runtime import reset_global_scheduler
from ecs_agent.types import (
    CompletionResult, DelegationCompletedEvent, EntityId, FinishReason, Message,
    RunCompletedEvent, SubagentConfig, ToolCall,
    FileRefPart, ImageUrlPart, StreamDelta,
)


@pytest.mark.parametrize(
    ("finish", "expected"),
    [
        (FinishReason.STOP, RunStatus.COMPLETED),
        (FinishReason.LENGTH, RunStatus.INCOMPLETE),
        (FinishReason.CONTENT_FILTER, RunStatus.INCOMPLETE),
        (FinishReason.CANCELLED, RunStatus.CANCELLED),
        (FinishReason.UNKNOWN, RunStatus.UNKNOWN),
        (FinishReason.ERROR, RunStatus.ERROR),
    ],
)
async def test_run_result_and_event_share_terminal_snapshot(finish: FinishReason, expected: RunStatus) -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(entity, ConversationComponent([Message(role="user", content="task")]))
    world.add_component(entity, LLMComponent(model=FakeModel([
        CompletionResult(message=Message(role="assistant", content="output"), finish_reason=finish)
    ])))
    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(ErrorHandlingSystem(), priority=99)
    events: list[RunCompletedEvent] = []

    async def record(event: RunCompletedEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(RunCompletedEvent, record)
    runner = Runner()
    result = await runner.run(world, max_ticks=3, target_entity=entity)
    assert result.status is expected
    assert runner.last_result is result
    assert len(events) == 1 and events[0].result is result
    assert result.success == (expected is RunStatus.COMPLETED)
    assert result.entity_id == entity
    if result.success:
        assert result.final_message is not None and result.final_message.content == "output"
        assert result.partial_message is None
    else:
        assert result.final_message is None
        assert result.partial_message is not None and result.partial_message.content == "output"


async def test_unrelated_entity_terminal_cannot_finish_target_run() -> None:
    world = World()
    target = world.create_entity()
    other = world.create_entity()
    world.add_component(target, ConversationComponent([]))

    class FinishTargetLater:
        """End an unrelated entity before the target produces a final result."""

        ticks = 0

        async def process(self, world: World) -> None:
            self.ticks += 1
            world.add_component(other, TerminalComponent(reason="reasoning_complete"))
            if self.ticks == 2:
                world.add_component(target, ConversationComponent([Message(
                    role="assistant", content="done", generation_status=GenerationStatus.COMPLETED,
                )]))
                world.add_component(target, TerminalComponent(reason="reasoning_complete"))

    system = FinishTargetLater()
    world.register_system(system, priority=0)
    result = await Runner().run(world, max_ticks=3, target_entity=target)
    assert system.ticks == 2
    assert result.success and result.entity_id == target


async def test_tick_budget_with_pending_tool_keeps_only_partial_result() -> None:
    world = World()
    entity = world.create_entity()
    tool = ToolCall(id="call_pending", name="unfinished", arguments={})
    world.add_component(entity, ConversationComponent([Message(
        role="assistant", content="working", tool_calls=[tool],
        generation_status=GenerationStatus.TOOL_CALLS,
    )]))
    world.add_component(entity, PendingToolCallsComponent([tool]))
    result = await Runner().run(world, max_ticks=1, target_entity=entity)
    assert result.status is RunStatus.MAX_TICKS and not result.success
    assert result.final_message is None
    assert result.partial_message is not None and result.partial_message.content == "working"


async def test_empty_completed_message_is_not_a_successful_delegation_result() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(entity, ConversationComponent([Message(
        role="assistant", content="", generation_status=GenerationStatus.COMPLETED,
    )]))
    world.add_component(entity, TerminalComponent(reason="reasoning_complete"))
    result = await Runner().run(world, max_ticks=1, target_entity=entity)
    assert not result.success and result.error is not None


@pytest.mark.parametrize(
    ("finish", "lifecycle"),
    [(FinishReason.LENGTH, "failed"), (FinishReason.CANCELLED, "cancelled"), (None, "cancelled")],
)
async def test_background_result_event_and_notification_agree(finish: FinishReason | None, lifecycle: str) -> None:
    reset_global_scheduler()
    world = World()
    parent = world.create_entity()
    started = asyncio.Event()

    class WaitingModel(FakeModel):
        """Wait until the enclosing task is externally cancelled."""

        async def complete(self, *args: Any, **kwargs: Any) -> CompletionResult:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    model = (WaitingModel if finish is None else FakeModel)([CompletionResult(
        message=Message(role="assistant", content="unfinished work"), finish_reason=finish,
    )])
    world.add_component(parent, SubagentRegistryComponent(subagents={
        "worker": SubagentConfig(name="worker", model=model),
    }))
    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    system = SubagentSystem()
    system.install_subagent_tool(world, parent)
    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None
    events: list[DelegationCompletedEvent] = []

    async def record(event: DelegationCompletedEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(DelegationCompletedEvent, record)
    try:
        launched = json.loads(await tools.handlers["subagent"](
            category="worker", prompt="complete task", load_skills=[], background=True, timeout=None,
        ))
        session_id = launched["session_id"]
        waiter = system._runtime_manager.get_or_create_session_event(session_id)
        task = await system._runtime_manager.get_task(session_id)
        assert task is not None
        if finish is None:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await asyncio.wait_for(task, timeout=2)
        assert waiter.is_set()
        session = await system._runtime_manager.get_session(session_id)
        assert session is not None and session.status == lifecycle
        expected_partial = "unfinished work" if finish is not None else None
        assert session.partial_content == expected_partial
        payload = json.loads(system._terminal_result_payload(session, "full", session_id))
        assert payload["status"] == "terminal"
        assert payload.get("partial_content") == expected_partial
        assert payload["outcome_status"] == session.outcome_status
        assert len(events) == 1 and not events[0].success
        assert events[0].outcome_status == session.outcome_status
        assert (events[0].partial_content or None) == session.partial_content
        table = world.get_component(parent, SubagentSessionTableComponent)
        assert table is not None and table.sessions[session_id].outcome_status == session.outcome_status
        queue = world.get_component(parent, SubagentNotificationQueueComponent)
        assert queue is not None
        assert queue.notifications[-1].terminal_status == lifecycle
    finally:
        reset_global_scheduler()


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_external_abort_preserves_partial_delegation_result(
    background: bool, stop: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_global_scheduler()
    world = World()
    parent = world.create_entity()
    waiting = asyncio.Event()

    class PartialModel(FakeModel):
        """Produce useful work, then wait for timeout or external cancellation."""

        async def _stream_complete(self, result: CompletionResult) -> AsyncIterator[StreamDelta]:
            yield StreamDelta(content="unfinished child work")
            waiting.set()
            await asyncio.Event().wait()

    world.add_component(parent, SubagentRegistryComponent(subagents={
        "worker": SubagentConfig(name="worker", model=PartialModel([
            CompletionResult(message=Message(role="assistant", content="unused")),
        ])),
    }))
    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    system = SubagentSystem(stream_subagents=True)
    system.install_subagent_tool(world, parent)
    registry = world.get_component(parent, ToolRegistryComponent)
    assert registry is not None
    events: list[DelegationCompletedEvent] = []
    runs: list[RunCompletedEvent] = []

    async def record(event: DelegationCompletedEvent) -> None:
        events.append(event)

    async def record_run(event: RunCompletedEvent) -> None:
        runs.append(event)

    assemble = system._assemble_child_world

    def assemble_and_observe(*args: Any, **kwargs: Any) -> tuple[World, EntityId]:
        child, target = assemble(*args, **kwargs)
        child.event_bus.subscribe(RunCompletedEvent, record_run)
        return child, target

    monkeypatch.setattr(system, "_assemble_child_world", assemble_and_observe)
    world.event_bus.subscribe(DelegationCompletedEvent, record)
    try:
        launch = asyncio.create_task(registry.handlers["subagent"](
            category="worker", prompt="task", load_skills=[], background=background,
            stream=True, timeout=0.1 if stop == "timeout" else None,
        ))
        session_id = None
        task = launch
        if background:
            session_id = json.loads(await launch)["session_id"]
            task = await system._runtime_manager.get_task(session_id)
            assert task is not None
        await asyncio.wait_for(waiting.wait(), timeout=2)
        if stop == "cancel":
            if session_id is not None:
                await system._runtime_manager.cancel_session(session_id)
            else:
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            text = await asyncio.wait_for(task, timeout=2)
            if not background:
                assert "Partial output (unfinished):\nunfinished child work" in text
        expected = RunStatus.TIMEOUT if stop == "timeout" else RunStatus.CANCELLED
        assert len(events) == 1
        event = events[0]
        assert not event.success and event.outcome_status == expected.value
        assert event.partial_content == "unfinished child work"
        assert event.run_result is not None and event.run_result.status is expected
        assert event.run_result.final_message is None
        assert len(runs) == 1 and runs[0].result.status is expected
        assert runs[0].result.partial_message.content == event.partial_content
        if session_id is not None:
            waiter = system._runtime_manager.get_or_create_session_event(session_id)
            assert waiter.is_set()
            session = await system._runtime_manager.get_session(session_id)
            assert session is not None
            payload = json.loads(system._terminal_result_payload(session, "full", session_id))
            assert payload["partial_content"] == event.partial_content
            assert payload["outcome_status"] == expected.value
            table = world.get_component(parent, SubagentSessionTableComponent)
            assert table.sessions[session_id].partial_content == event.partial_content
    finally:
        reset_global_scheduler()


async def test_nontext_final_result_survives_legacy_delegation_boundary() -> None:
    world = World()
    entity = world.create_entity()
    message = Message(role="assistant", content="", generation_status=GenerationStatus.COMPLETED,
                      parts=[FileRefPart(file_id="file_output", filename="report.pdf"),
                             ImageUrlPart(url="https://example.invalid/image.png")])
    world.add_component(entity, ConversationComponent([message]))
    world.add_component(entity, TerminalComponent(reason="reasoning_complete"))
    result = await Runner().run(world, max_ticks=1, target_entity=entity)
    delegated = DelegationResult.from_run(result)
    text, success, error = delegated.as_tuple()
    assert success and error is None
    assert json.loads(text) == {"content": "", "parts": [
        {"type": "file", "file_id": "file_output", "filename": "report.pdf"},
        {"type": "image_url", "url": "https://example.invalid/image.png", "detail": None},
    ]}


async def test_old_assistant_before_new_user_is_not_a_current_final_result() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(entity, ConversationComponent([
        Message(role="assistant", content="old answer", generation_status=GenerationStatus.COMPLETED),
        Message(role="user", content="new task"),
    ]))
    world.add_component(entity, TerminalComponent(reason="reasoning_complete"))
    result = await Runner().run(world, max_ticks=1, target_entity=entity)
    assert not result.success and result.final_message is None


@pytest.mark.parametrize("operation", ["reasoning", "planning", "replanning"])
async def test_context_budget_exhaustion_survives_error_cleanup(operation: str) -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=FakeModel([])))
    world.add_component(entity, ConversationComponent([
        Message(role="assistant", content="prior partial work", generation_status=GenerationStatus.COMPLETED),
        Message(role="user", content="protected new task " * 50),
    ]))
    world.add_component(entity, PlanComponent(steps=["first", "next"], current_step=1))
    world.add_component(entity, ContextTrimConfig(max_tokens=1, overflow_behavior="error"))
    system = {"reasoning": ReasoningSystem(), "planning": PlanningSystem(), "replanning": ReplanningSystem()}[operation]
    world.register_system(system, priority=0)
    world.register_system(ErrorHandlingSystem(), priority=99)
    result = await Runner().run(world, max_ticks=2, target_entity=entity)
    assert result.status is RunStatus.BUDGET_EXHAUSTED and not result.success
    assert result.error == "Protected content exceeds configured budget"
    assert not world.has_component(entity, ErrorComponent)
    delegated = DelegationResult.from_run(result)
    text, success, error = delegated.as_tuple()
    assert not success and error == result.error and "prior partial work" in text
