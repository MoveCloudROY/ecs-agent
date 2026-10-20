"""Behavioral acceptance of entity-bound turns, continuation, and recovery."""

import asyncio

import pytest

from ecs_agent import AgentHandle
from ecs_agent.components import (
    AgentTurnComponent, ConversationComponent, LLMComponent, PendingToolCallsComponent,
    TerminalComponent, ToolApprovalComponent,
)
from ecs_agent.core import Runner, RunStatus, World
from ecs_agent.history import replace_conversation_history
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.serialization import WorldSerializer
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.terminal_cleanup import TerminalCleanupSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools import register_tools, tool
from ecs_agent.types import (
    ApprovalPolicy, CompletionResult, Message, RunCompletedEvent, StreamContentDeltaEvent,
    ToolApprovalRequestedEvent, ToolCall,
)


def reply(text: str) -> CompletionResult:
    return CompletionResult(message=Message(role="assistant", content=text))


def make_agent(responses: list[CompletionResult]) -> tuple[World, AgentHandle, FakeModel]:
    world = World()
    entity = world.create_entity()
    model = FakeModel(responses)
    world.add_component(entity, LLMComponent(model=model))
    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(TerminalCleanupSystem(), priority=1)
    world.register_system(ToolExecutionSystem(), priority=5)
    return world, AgentHandle(world, entity), model


async def test_two_inputs_create_two_turns_without_repeating_completed_agent() -> None:
    world, agent, model = make_agent([reply("one"), reply("two")])
    first = await agent.run_turn("first")
    assert first.success and first.ticks == 1
    assert first.final_message is not None and first.final_message.content == "one"
    assert world.has_component(agent.entity, TerminalComponent)
    await world.process()
    assert model._index == 1
    second = await agent.run_turn("second")
    assert second.success and second.ticks == 1
    assert first.turn_id != second.turn_id
    assert first.run_id != second.run_id
    assert second.final_message is not None and second.final_message.content == "two"
    assert first.final_message.content == "one"
    history = world.get_component(agent.entity, ConversationComponent)
    assert history is not None
    assert [message.content for message in history.messages] == ["first", "one", "second", "two"]
    assert [message.turn_id for message in history.messages] == [first.turn_id, first.turn_id, second.turn_id, second.turn_id]
    assert AgentHandle(world, agent.entity).last_result is second


async def test_zero_budget_then_resume_submits_input_only_once() -> None:
    world, agent, model = make_agent([reply("answer")])
    paused = await agent.run_turn("question", max_steps=0)
    assert paused.status is RunStatus.MAX_TICKS
    assert paused.ticks == model._index == 0
    assert paused.partial_message is None
    resumed = await agent.resume(max_steps=1)
    assert resumed.success and resumed.turn_id == paused.turn_id
    assert resumed.run_id != paused.run_id
    history = world.get_component(agent.entity, ConversationComponent)
    assert history is not None
    assert [message.role for message in history.messages] == ["user", "assistant"]


async def test_user_input_event_is_emitted_once_and_identifies_the_submitting_run() -> None:
    from ecs_agent.types import UserInputReceivedEvent

    world, agent, _ = make_agent([reply("answer"), reply("next answer")])
    received = []

    async def observe(event: UserInputReceivedEvent) -> None:
        received.append(event)

    world.event_bus.subscribe(UserInputReceivedEvent, observe)
    first = await agent.run_turn("first", max_steps=0)
    assert (await agent.resume()).success
    second = await agent.run_turn("second")
    assert [event.text for event in received] == ["first", "second"]
    assert [event.turn_id for event in received] == [first.turn_id, second.turn_id]
    assert [event.run_id for event in received] == [first.run_id, second.run_id]
    assert all(event.source == "agent_handle" for event in received)


async def test_new_turn_without_output_does_not_reuse_previous_answer() -> None:
    _, agent, _ = make_agent([reply("old")])
    await agent.run_turn("first")
    result = await agent.run_turn("second", max_steps=0)
    assert result.partial_message is result.final_message is None


async def test_reused_provider_message_does_not_reassign_previous_turn() -> None:
    response = reply("same response object")
    world, agent, _ = make_agent([response, response])
    first = await agent.run_turn("first")
    second = await agent.run_turn("second")
    assert first.success and second.success
    history = world.get_component(agent.entity, ConversationComponent)
    assert history is not None
    assert [message.turn_id for message in history.messages] == [first.turn_id, first.turn_id, second.turn_id, second.turn_id]


@pytest.mark.parametrize("terminal", [False, True])
async def test_scope_does_not_process_another_idle_or_terminal_entity(terminal: bool) -> None:
    world, agent, _ = make_agent([reply("target")])
    other = world.create_entity()
    other_model = FakeModel([reply("unwanted")])
    world.add_component(other, LLMComponent(model=other_model))
    world.add_component(other, ConversationComponent([Message(role="user", content="do not run")]))
    if terminal:
        world.add_component(other, TerminalComponent(reason="reasoning_complete"))
    result = await agent.run_turn("run target")
    assert result.success
    assert other_model._index == 0
    assert world.has_component(other, TerminalComponent) is terminal


async def test_completed_tools_are_not_replayed_on_budget_resume() -> None:
    calls: list[int] = []

    @tool()
    async def total(values: list[int]) -> str:
        calls.append(sum(values))
        return str(calls[-1])

    world, agent, _ = make_agent([
        CompletionResult(message=Message(role="assistant", content="", tool_calls=[ToolCall("one", "total", {"values": [1, 2]})])),
        reply("3"),
    ])
    register_tools(world, agent.entity, [total])
    first = await agent.run_turn("sum", max_steps=1)
    assert first.status is RunStatus.MAX_TICKS
    assert calls == [3]
    assert not world.has_component(agent.entity, PendingToolCallsComponent)
    second = await agent.resume(max_steps=1)
    assert second.success and calls == [3]


@pytest.mark.parametrize("recording", [False, True])
async def test_cancelled_tool_with_unknown_outcome_blocks_resume_and_checkpoint(tmp_path, recording: bool) -> None:
    entered = asyncio.Event()
    count = 0

    @tool()
    async def write() -> str:
        nonlocal count
        count += 1
        entered.set()
        await asyncio.Event().wait()
        return "written"

    world, agent, model = make_agent([
        CompletionResult(message=Message(role="assistant", content="", tool_calls=[ToolCall("write-one", "write", {})])),
    ])
    registry = register_tools(world, agent.entity, [write])
    if recording:
        from ecs_agent.sessions import InMemorySessionStore, SessionService, install_session_recording

        install_session_recording(
            world, agent.entity, SessionService(InMemorySessionStore()),
            workspace_id="test", agent_name="writer", register_tools=False,
        )
    task = asyncio.create_task(agent.run_turn("write"))
    await asyncio.wait_for(entered.wait(), timeout=2)
    in_flight = tmp_path / "in-flight.json"
    Runner().save_checkpoint(world, in_flight)
    snapshot, _ = Runner.load_checkpoint(in_flight, {"fake": model}, registry.handlers)
    assert snapshot.get_component(agent.entity, AgentTurnComponent).status == "blocked"
    await agent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    state = world.get_component(agent.entity, AgentTurnComponent)
    assert state is not None and state.status == "blocked"
    assert agent.last_result is not None and agent.last_result.status is RunStatus.CANCELLED
    assert state.unconfirmed_tool_calls == ["write-one"]
    with pytest.raises(ValueError, match="cannot be resumed"):
        await agent.resume()
    checkpoint = tmp_path / "agent.json"
    Runner().save_checkpoint(world, checkpoint)
    restored, _ = Runner.load_checkpoint(checkpoint, {"fake": model}, registry.handlers)
    restored_state = restored.get_component(agent.entity, AgentTurnComponent)
    assert restored_state is not None and restored_state.status == "blocked"
    if recording:
        from ecs_agent.sessions import release_session_writer

        service = world.session_service
        assert service is not None
        release_session_writer(world, agent.entity)
        install_session_recording(
            restored, agent.entity, service, workspace_id="test",
            agent_name="writer", register_tools=False, restore=True,
        )
    with pytest.raises(ValueError, match="cannot be resumed"):
        await AgentHandle(restored, agent.entity).resume()
    assert count == 1


async def test_cancel_while_awaiting_approval_can_resume_without_new_input() -> None:
    entered = asyncio.Event()
    count = 0
    requests = 0

    @tool()
    async def write() -> str:
        nonlocal count
        count += 1
        return "written"

    world, agent, _ = make_agent([
        CompletionResult(message=Message(role="assistant", content="", tool_calls=[ToolCall("approval-one", "write", {})])),
        reply("done"),
    ])
    register_tools(world, agent.entity, [write])
    world.add_component(agent.entity, ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL))

    async def approve(event: ToolApprovalRequestedEvent) -> None:
        nonlocal requests
        requests += 1
        entered.set()
        if requests == 1:
            await asyncio.Event().wait()
        event.approval_future.set_result(True)

    world.event_bus.subscribe(ToolApprovalRequestedEvent, approve)
    task = asyncio.create_task(agent.run_turn("write"))
    await asyncio.wait_for(entered.wait(), timeout=2)
    await agent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await agent.resume()
    assert result.success and count == 1 and requests == 2


async def test_checkpoint_preserves_completed_turn_and_new_turn_is_explicit(tmp_path) -> None:
    world, agent, model = make_agent([reply("first"), reply("second")])
    first = await agent.run_turn("one")
    checkpoint = tmp_path / "complete.json"
    Runner().save_checkpoint(world, checkpoint)
    restored, _ = Runner.load_checkpoint(checkpoint, {"fake": model}, {})
    restored.register_system(ReasoningSystem(), priority=0)
    handle = AgentHandle(restored, agent.entity)
    with pytest.raises(ValueError, match="cannot be resumed"):
        await handle.resume()
    result = await handle.run_turn("two")
    assert result.success and result.turn_id != first.turn_id
    assert model._index == 2


async def test_checkpoint_preserves_media_in_the_current_turn_result(tmp_path) -> None:
    from ecs_agent.types import FileRefPart, ImageUrlPart

    response = CompletionResult(message=Message(role="assistant", content="", parts=[
        FileRefPart(file_id="file-result", filename="report.txt"),
        ImageUrlPart(url="https://example.invalid/result.png"),
    ]))
    world, agent, model = make_agent([response])
    result = await agent.run_turn("make a report")
    assert result.success
    path = tmp_path / "media.json"
    Runner().save_checkpoint(world, path)
    restored, _ = Runner.load_checkpoint(path, {"fake": model}, {})
    snapshot = await Runner().run(restored, target_entity=agent.entity)
    assert snapshot.success and snapshot.final_message == result.final_message


async def test_current_turn_output_survives_working_history_compaction() -> None:
    world, agent, _ = make_agent([reply("answer")])
    result = await agent.run_turn("question")
    replace_conversation_history(world, agent.entity, [], reason="compaction")
    snapshot = await Runner().run(world, target_entity=agent.entity)
    assert snapshot.turn_id == result.turn_id
    assert snapshot.final_message is not None and snapshot.final_message.content == "answer"


def test_old_checkpoint_requires_explicit_adoption() -> None:
    world, agent, model = make_agent([reply("answer")])
    payload = WorldSerializer.to_dict(world)
    payload["schema_version"] = 2
    restored = WorldSerializer.from_dict(payload, {"fake": model}, {})
    handle = AgentHandle(restored, agent.entity)
    state = restored.get_component(agent.entity, AgentTurnComponent)
    assert state is not None and state.status == "legacy"
    assert handle.adopt_turn()
    assert restored.get_component(agent.entity, AgentTurnComponent).status == "paused"


def test_native_tree_is_rejected_before_silently_losing_user_input() -> None:
    from ecs_agent.components import ConversationTreeComponent

    world, agent, _ = make_agent([])
    world.add_component(agent.entity, ConversationTreeComponent())
    with pytest.raises(ValueError, match="native tree writes"):
        AgentHandle(world, agent.entity)


async def test_stream_events_share_turn_and_run_and_include_final_result() -> None:
    _, agent, _ = make_agent([reply("answer")])
    received = []
    async with agent.stream_turn("question") as events:
        async for event in events:
            received.append(event)
    assert received
    completed = [event.event for event in received if isinstance(event.event, RunCompletedEvent)]
    assert len(completed) == 1 and completed[0].result.success
    result = completed[0].result
    assert {event.entity_id for event in received} == {agent.entity}
    assert {event.turn_id for event in received} == {result.turn_id}
    assert {event.run_id for event in received} == {result.run_id}
    assert "".join(event.event.delta for event in received if isinstance(event.event, StreamContentDeltaEvent)) == "answer"


async def test_stream_does_not_attribute_delayed_events_from_a_previous_run() -> None:
    from ecs_agent.observability.context import reset_run_context, set_run_context
    from ecs_agent.types import StreamStartEvent

    world, agent, _ = make_agent([reply("answer")])

    async def delayed_old_event(event: StreamStartEvent) -> None:
        token = set_run_context(trace_id="old-trace", run_id="old-run")
        try:
            await world.event_bus.publish(StreamContentDeltaEvent(entity_id=agent.entity, delta="STALE"))
        finally:
            reset_run_context(token)

    world.event_bus.subscribe(StreamStartEvent, delayed_old_event)
    content = []
    async with agent.stream_turn("question") as events:
        async for envelope in events:
            if isinstance(envelope.event, StreamContentDeltaEvent):
                content.append(envelope.event.delta)
    assert "".join(content) == "answer"


async def test_scoped_stream_orders_final_result_after_all_deltas() -> None:
    from ecs_agent.components import StreamingComponent

    world, agent, _ = make_agent([reply("answer")])
    original = StreamingComponent(enabled=True, non_blocking_delta_publish=True)
    world.add_component(agent.entity, original)

    async def slow_subscriber(event: StreamContentDeltaEvent) -> None:
        await asyncio.sleep(0.01)

    world.event_bus.subscribe(StreamContentDeltaEvent, slow_subscriber)
    received = []
    async with agent.stream_turn("question") as events:
        async for envelope in events:
            received.append(envelope.event)
    assert "".join(event.delta for event in received if isinstance(event, StreamContentDeltaEvent)) == "answer"
    assert isinstance(received[-1], RunCompletedEvent)
    assert world.get_component(agent.entity, StreamingComponent) is original


async def test_stream_context_exit_cancels_and_releases_pending_run() -> None:
    from ecs_agent.types import StreamDelta

    class SlowStream(FakeModel):
        async def _stream_complete(self, result):
            yield StreamDelta(content="partial")
            await asyncio.Event().wait()

    world, agent, _ = make_agent([])
    world.add_component(agent.entity, LLMComponent(model=SlowStream([reply("unused")])))
    async with agent.stream_turn("question") as events:
        async for event in events:
            if isinstance(event.event, StreamContentDeltaEvent):
                break
    assert world._agent_run is None
    assert world._runner_task is None
    assert agent.last_result is not None and agent.last_result.status is RunStatus.CANCELLED
    state = world.get_component(agent.entity, AgentTurnComponent)
    assert state is not None and state.status == "paused"
    world.add_component(agent.entity, LLMComponent(model=FakeModel([reply("continued")])))
    async with agent.stream_resume() as events:
        async for _ in events:
            pass
    assert agent.last_result.success


async def test_cancelling_driver_before_first_tick_releases_world() -> None:
    world, agent, model = make_agent([reply("later")])
    task = agent._start("question", max_steps=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert world._agent_run is None
    assert model._index == 0
    assert agent.last_result is not None and agent.last_result.status is RunStatus.CANCELLED
    assert (await agent.resume()).success


async def test_checkpoint_between_tool_results_never_replays_completed_prefix(tmp_path) -> None:
    from ecs_agent.types import ToolExecutionCompletedEvent

    calls = []

    @tool()
    async def write(value: int) -> str:
        calls.append(value)
        return str(value)

    world, agent, _ = make_agent([
        CompletionResult(message=Message(role="assistant", content="", tool_calls=[
            ToolCall("first", "write", {"value": 1}), ToolCall("second", "write", {"value": 2}),
        ])),
        reply("done"),
    ])
    registry = register_tools(world, agent.entity, [write])
    checkpoint = tmp_path / "between-results.json"

    async def save(event: ToolExecutionCompletedEvent) -> None:
        if event.tool_call_id == "first":
            Runner().save_checkpoint(world, checkpoint)

    world.event_bus.subscribe(ToolExecutionCompletedEvent, save)
    await agent.run_turn("write")
    assert calls == [1, 2]
    calls.clear()
    resumed_model = FakeModel([reply("resumed")])
    restored, _ = Runner.load_checkpoint(checkpoint, {"fake": resumed_model}, registry.handlers)
    restored.register_system(ReasoningSystem(), priority=0)
    restored.register_system(ToolExecutionSystem(), priority=5)
    result = await AgentHandle(restored, agent.entity).resume()
    assert result.success
    assert calls == [2]


async def test_new_input_recording_failure_preserves_previous_turn(monkeypatch) -> None:
    from ecs_agent.sessions import InMemorySessionStore, SessionService, install_session_recording

    world, agent, _ = make_agent([reply("first")])
    service = SessionService(InMemorySessionStore())
    install_session_recording(world, agent.entity, service, workspace_id="test", agent_name="agent", register_tools=False)
    first = await agent.run_turn("one")
    previous = world.get_component(agent.entity, AgentTurnComponent)
    conversation = world.get_component(agent.entity, ConversationComponent)
    assert conversation is not None
    messages = list(conversation.messages)

    def fail(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(service, "append_message", fail)
    with pytest.raises(OSError, match="storage unavailable"):
        await agent.run_turn("two")
    assert world.get_component(agent.entity, AgentTurnComponent) is previous
    assert conversation.messages == messages
    assert world.has_component(agent.entity, TerminalComponent)
    assert agent.last_result is first
    assert world._agent_run is world._runner_task is None


async def test_concurrent_handle_and_manual_tick_cannot_duplicate_inference() -> None:
    entered = asyncio.Event()

    class WaitingModel(FakeModel):
        async def complete(self, *args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    world, agent, _ = make_agent([])
    world.add_component(agent.entity, LLMComponent(model=WaitingModel([])))
    task = asyncio.create_task(agent.run_turn("one"))
    await asyncio.wait_for(entered.wait(), timeout=2)
    try:
        with pytest.raises(RuntimeError, match="already driving"):
            await AgentHandle(world, agent.entity).run_turn("two")
        with pytest.raises(RuntimeError, match="active execution driver"):
            await world.process()
        conversation = world.get_component(agent.entity, ConversationComponent)
        assert conversation is not None
        assert [message.content for message in conversation.messages] == ["one"]
    finally:
        await agent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_scope_keeps_unrelated_pending_tools_and_approvals_untouched() -> None:
    world, agent, _ = make_agent([reply("target")])
    other = world.create_entity()
    called = []

    @tool()
    async def probe() -> str:
        called.append(True)
        return "unexpected"

    register_tools(world, other, [probe])
    world.add_component(other, ConversationComponent([]))
    pending = PendingToolCallsComponent([ToolCall("other", "probe", {})])
    world.add_component(other, pending)
    world.add_component(other, ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_DENY))
    from ecs_agent.systems.tool_approval import ToolApprovalSystem

    world.register_system(ToolApprovalSystem(), priority=1)
    result = await agent.run_turn("only target")
    assert result.success and not called
    assert world.get_component(other, PendingToolCallsComponent) is pending
    assert world.get_component(other, ConversationComponent).messages == []


async def test_failed_turn_requires_explicit_abandon_before_new_input() -> None:
    world, agent, model = make_agent([])
    failed = await agent.run_turn("one")
    assert not failed.success
    with pytest.raises(ValueError, match="Previous turn"):
        await agent.run_turn("two")
    agent.abandon_turn(reason="Use a new request after provider recovery")
    assert agent.last_result is failed and not failed.success
    model._responses.append(reply("recovered"))
    assert (await agent.run_turn("two")).success


@pytest.mark.parametrize("budget", [-1, True, 1.5])
async def test_invalid_per_run_budget_does_not_submit_input(budget) -> None:
    world, agent, model = make_agent([reply("unused")])
    with pytest.raises(ValueError, match="max_steps"):
        await agent.run_turn("must not commit", max_steps=budget)
    assert world.get_component(agent.entity, AgentTurnComponent) is None
    assert world.get_component(agent.entity, ConversationComponent) is None
    assert model._index == 0


async def test_scoped_parent_can_delegate_to_its_child_world() -> None:
    from ecs_agent.components import MessageBusConfigComponent, SubagentRegistryComponent, ToolRegistryComponent
    from ecs_agent.systems.message_bus import MessageBusSystem
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.types import SubagentConfig

    child = FakeModel([reply("child answer")])
    world, agent, _ = make_agent([
        CompletionResult(message=Message(role="assistant", content="", tool_calls=[
            ToolCall("delegate", "subagent", {"category": "helper", "prompt": "answer the question"}),
        ])),
        reply("parent answer"),
    ])
    world.add_component(agent.entity, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(agent.entity, SubagentRegistryComponent(subagents={
        "helper": SubagentConfig(name="helper", model=child, system_prompt="Answer briefly"),
    }))
    world.add_component(agent.entity, MessageBusConfigComponent())
    world.register_system(MessageBusSystem(), priority=-2)
    world.register_system(SubagentSystem(), priority=-1)
    result = await agent.run_turn("delegate the question", max_steps=5)
    assert result.success
    assert child._index == 1
    conversation = world.get_component(agent.entity, ConversationComponent)
    assert conversation is not None
    assert [message.content for message in conversation.messages if message.role == "tool"] == ["child answer"]


async def test_checkpoint_resume_requires_new_approval_for_unstarted_call(tmp_path) -> None:
    from ecs_agent.systems.tool_approval import ToolApprovalSystem

    effects = []

    @tool()
    async def probe() -> str:
        effects.append("executed")
        return "ok"

    model = FakeModel([
        CompletionResult(message=Message(role="assistant", content="", tool_calls=[ToolCall("approved", "probe", {})])),
        reply("done"),
    ])
    # Keep the first run at the preflight phase, before actual execution.
    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(entity, ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL))
    registry = register_tools(world, entity, [probe])
    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(ToolApprovalSystem(), priority=1)
    requests = []

    async def approve(event: ToolApprovalRequestedEvent) -> None:
        requests.append(event.tool_call.id)
        event.approval_future.set_result(True)

    world.event_bus.subscribe(ToolApprovalRequestedEvent, approve)
    first = await AgentHandle(world, entity).run_turn("probe", max_steps=1)
    assert first.status is RunStatus.MAX_TICKS and requests == ["approved"] and not effects
    path = tmp_path / "approved.json"
    Runner().save_checkpoint(world, path)
    restored, _ = Runner.load_checkpoint(path, {"fake": model}, registry.handlers)
    restored.event_bus.subscribe(ToolApprovalRequestedEvent, approve)
    restored.register_system(ReasoningSystem(), priority=0)
    restored.register_system(ToolExecutionSystem(), priority=5)
    result = await AgentHandle(restored, entity).resume()
    assert result.success and requests == ["approved", "approved"] and effects == ["executed"]


async def test_changed_call_id_after_start_cannot_requeue_completed_effect(tmp_path) -> None:
    from ecs_agent.types import ToolExecutionCompletedEvent, ToolExecutionStartedEvent

    effects = []

    @tool()
    async def probe() -> str:
        effects.append("effect")
        await asyncio.sleep(0)
        return "ok"

    world, agent, _ = make_agent([
        CompletionResult(message=Message(role="assistant", content="", tool_calls=[ToolCall("original", "probe", {})])),
        reply("done"),
    ])
    registry = register_tools(world, agent.entity, [probe])
    path = tmp_path / "after-effect.json"

    async def change(event: ToolExecutionStartedEvent) -> None:
        pending = world.get_component(agent.entity, PendingToolCallsComponent)
        pending.tool_calls[0].id = "changed-after-start"

    async def checkpoint(event: ToolExecutionCompletedEvent) -> None:
        Runner().save_checkpoint(world, path)

    world.event_bus.subscribe(ToolExecutionStartedEvent, change)
    world.event_bus.subscribe(ToolExecutionCompletedEvent, checkpoint)
    assert (await agent.run_turn("probe")).success
    restored_model = FakeModel([reply("continued")])
    restored, _ = Runner.load_checkpoint(path, {"fake": restored_model}, registry.handlers)
    restored.register_system(ReasoningSystem(), priority=0)
    restored.register_system(ToolExecutionSystem(), priority=5)
    assert (await AgentHandle(restored, agent.entity).resume()).success
    assert effects == ["effect"]
