"""End-to-end recording of model turns, tools, and semantic run outcomes."""

import asyncio

import pytest

from ecs_agent.components import (
    ConversationComponent, ErrorComponent, LLMComponent, PendingToolCallsComponent,
    SessionPolicyComponent, SessionRefComponent, TerminalComponent, ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.history import append_conversation_message
from ecs_agent.providers import FakeModel
from ecs_agent.scratchbook.artifact_registry import ArtifactRegistry
from ecs_agent.sessions import InMemorySessionStore, SessionService, install_session_recording
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message, ToolCall, ToolExecutionCompletedEvent, ToolSchema


def setup(tmp_path, handler, *, responses=None):
    world = World()
    entity = world.create_entity()
    call = ToolCall(id="call_evidence", name="execute", arguments={})
    world.add_component(entity, ConversationComponent(messages=[Message(role="user", content="do test")]))
    world.add_component(entity, ToolRegistryComponent(
        tools={"execute": ToolSchema(name="execute", description="test", parameters={"type": "object"})},
        handlers={"execute": handler},
    ))
    if responses:
        world.add_component(entity, LLMComponent(model=FakeModel(responses)))
    registry = ArtifactRegistry(tmp_path)
    service = SessionService(InMemorySessionStore(), artifact_registry=registry)
    install_session_recording(world, entity, service, workspace_id="project", agent_name="coder", register_tools=False)
    return world, entity, call, service, registry


async def test_model_tool_and_terminal_are_committed_before_observation(tmp_path):
    evidence = "唯一完整证据" * 1000

    async def handler():
        ref = world.get_component(entity, SessionRefComponent)
        assert any(r.kind == "tool_started" for r in service.store.records(ref.session_id))
        return evidence

    call = ToolCall(id="call_evidence", name="execute", arguments={})
    world, entity, _, service, registry = setup(tmp_path, handler, responses=[
        CompletionResult(message=Message(role="assistant", content="run test", tool_calls=[call])),
        CompletionResult(message=Message(role="assistant", content="review complete")),
    ])
    observed = []

    async def on_completed(event):
        ref = world.get_component(entity, SessionRefComponent)
        hits = service.search(ref.session_id, SessionPolicyComponent(), "唯一完整证据").items
        observed.append(len(hits))

    world.event_bus.subscribe(ToolExecutionCompletedEvent, on_completed)
    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(ToolExecutionSystem(registry=registry), priority=1)
    result = await Runner().run(world, target_entity=entity)
    ref = world.get_component(entity, SessionRefComponent)
    records = list(service.store.records(ref.session_id))
    assert result.success
    assert observed == [1]
    assert service.store.get_session(ref.session_id).status == "completed"
    assert sum(r.payload.get("content") == "review complete" for r in records) == 1
    assert sum(r.payload.get("role") == "tool" for r in records) == 1
    assert len(service.search(ref.session_id, SessionPolicyComponent(), "review complete").items) == 1
    assert records[-1].kind == "run_finished"
    service.close()


def test_prompt_context_is_recorded_before_queue_consumption(tmp_path, monkeypatch):
    from ecs_agent.components import ContextEntry, PromptContextQueueComponent, PromptContextReservationComponent
    from ecs_agent.prompts.message_assembly import commit_prompt_context_reservation

    async def handler():
        return "unused"

    world, entity, _, service, _ = setup(tmp_path, handler)
    entry = ContextEntry(entry_id="entry", source_label="test", content="injected evidence", registration_order=0, priority=0)
    queue = PromptContextQueueComponent(entries=[entry])
    reservation = PromptContextReservationComponent(reservation_id="reservation", created_at_tick=0, reserved_entries=[entry], rendered_block="injected evidence")
    conversation = world.get_component(entity, ConversationComponent)
    original = service.store.append

    def fail(*args, **kwargs):
        raise OSError("cannot commit context")

    monkeypatch.setattr(service.store, "append", fail)
    with pytest.raises(OSError):
        commit_prompt_context_reservation(queue=queue, reservation=reservation, conversation=conversation, world=world, entity_id=entity)
    assert queue.entries == [entry]
    assert len(conversation.messages) == 1
    monkeypatch.setattr(service.store, "append", original)
    commit_prompt_context_reservation(queue=queue, reservation=reservation, conversation=conversation, world=world, entity_id=entity)
    assert queue.entries == []
    ref = world.get_component(entity, SessionRefComponent)
    assert len(service.search(ref.session_id, SessionPolicyComponent(), "injected evidence").items) == 1
    service.close()


async def test_result_commit_failure_never_reexecutes_external_tool(tmp_path, monkeypatch):
    executed = []

    async def handler():
        executed.append("side effect")
        return "result"

    world, entity, call, service, _ = setup(tmp_path, handler)
    world.add_component(entity, PendingToolCallsComponent(tool_calls=[call]))
    original = service.store.append

    def fail_result(draft, **kwargs):
        if draft.payload.get("role") == "tool":
            raise OSError("disk failed after execution")
        return original(draft, **kwargs)

    monkeypatch.setattr(service.store, "append", fail_result)
    system = ToolExecutionSystem()
    with pytest.raises(OSError):
        await system.process(world)
    assert executed == ["side effect"]
    assert world.get_component(entity, ErrorComponent) is not None
    assert world.get_component(entity, PendingToolCallsComponent) is None
    await system.process(world)
    assert executed == ["side effect"]
    ref = world.get_component(entity, SessionRefComponent)
    records = list(service.store.records(ref.session_id))
    assert any(r.kind == "tool_started" for r in records)
    assert not any(r.payload.get("role") == "tool" for r in records)
    service.close()


async def test_cancellation_records_unknown_tool_completion_without_rerun(tmp_path):
    entered = asyncio.Event()
    executions = []

    async def handler():
        executions.append(1)
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    world, entity, call, service, _ = setup(tmp_path, handler)
    world.add_component(entity, PendingToolCallsComponent(tool_calls=[call]))
    system = ToolExecutionSystem()
    task = asyncio.create_task(system.process(world))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert world.get_component(entity, PendingToolCallsComponent) is None
    await system.process(world)
    assert executions == [1]
    ref = world.get_component(entity, SessionRefComponent)
    assert any(r.kind == "tool_started" for r in service.store.records(ref.session_id))
    service.close()


async def test_reintroduced_pending_call_is_not_treated_as_unexecuted(tmp_path):
    executions = []

    async def handler():
        executions.append(1)
        return "done"

    world, entity, call, service, _ = setup(tmp_path, handler)
    world.add_component(entity, PendingToolCallsComponent(tool_calls=[call]))
    system = ToolExecutionSystem()
    await system.process(world)
    world.add_component(entity, PendingToolCallsComponent(tool_calls=[call]))
    with pytest.raises(RuntimeError, match="already|confirmed"):
        await system.process(world)
    assert executions == [1]
    service.close()


async def test_terminal_record_failure_cannot_leave_successful_run_result(tmp_path, monkeypatch):
    from ecs_agent.core.run_result import RunStatus

    async def handler():
        return "unused"

    world, entity, _, service, _ = setup(tmp_path, handler, responses=[CompletionResult(message=Message(role="assistant", content="final answer"))])
    original = service.store.append

    def fail_terminal(draft, **kwargs):
        if draft.kind == "run_finished":
            raise OSError("terminal not persisted")
        return original(draft, **kwargs)

    monkeypatch.setattr(service.store, "append", fail_terminal)
    world.register_system(ReasoningSystem(), priority=0)
    runner = Runner()
    with pytest.raises(OSError):
        await runner.run(world, target_entity=entity)
    assert runner.last_result.status is RunStatus.ERROR
    assert runner.last_result.final_message is None
    assert world.get_component(entity, ErrorComponent) is not None
    service.close()


async def test_transient_user_render_is_recorded_once_before_observation(tmp_path):
    from ecs_agent.components import UserPromptConfigComponent
    from ecs_agent.prompts.contracts import TriggerSpec
    from ecs_agent.systems.user_prompt_normalization_system import UserPromptNormalizationSystem
    from ecs_agent.types import PromptReplacementEvent

    async def handler():
        return "unused"

    world, entity, _, service, _ = setup(tmp_path, handler)
    world.add_component(entity, UserPromptConfigComponent(triggers=[TriggerSpec(pattern="do test", match_mode="exact", action="replace", content="unique rendered input")]))
    seen = []

    async def observe(event):
        ref = world.get_component(entity, SessionRefComponent)
        seen.append(len(service.search(ref.session_id, SessionPolicyComponent(), "unique rendered input").items))

    world.event_bus.subscribe(PromptReplacementEvent, observe)
    system = UserPromptNormalizationSystem()
    await system.process(world)
    await system.process(world)
    assert seen == [1]
    ref = world.get_component(entity, SessionRefComponent)
    assert sum(record.kind == "prompt_render" for record in service.store.records(ref.session_id)) == 1
    service.close()
