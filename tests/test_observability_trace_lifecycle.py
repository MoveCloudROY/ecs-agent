"""Observability EventBus subscriber trace lifecycle tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from ecs_agent.components.definitions import (
    ConversationComponent,
    LLMComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.observability import (
    LLMObservationCompletedEvent,
    LLMObservationStartedEvent,
    RecordingTelemetrySink,
    current_run_id,
    current_trace_id,
    install_observability,
    reset_run_context,
    set_run_context,
)
from ecs_agent.observability.schema import TelemetryRecord
from ecs_agent.types import (
    CompletionResult,
    ErrorOccurredEvent,
    RunCompletedEvent,
    RunnerTickCompletedEvent,
    RunnerTickStartedEvent,
    RunStartedEvent,
    SystemExecutionCompletedEvent,
    SystemExecutionStartedEvent,
    Message,
    SubagentConfig,
    Usage,
    UserInputReceivedEvent,
    WorkflowStateEvaluatedEvent,
)
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.systems.subagent import SubagentSystem


class TerminatingSystem:
    """System that terminates a run on its first tick."""

    async def process(self, world: World) -> None:
        """Attach a terminal component."""
        world.add_component(world.create_entity(), TerminalComponent(reason="done"))


class ErrorEventSystem:
    """System that emits an error event before terminating."""

    def __init__(self) -> None:
        self.entity_id: int | None = None

    async def process(self, world: World) -> None:
        """Publish an error event and attach a terminal component."""
        entity_id = world.create_entity()
        self.entity_id = int(entity_id)
        await world.event_bus.publish(
            ErrorOccurredEvent(
                entity_id=entity_id,
                error="recoverable failure",
                system_name="ErrorEventSystem",
            )
        )
        world.add_component(entity_id, TerminalComponent(reason="done"))


class RaisingSystem:
    """System that raises a normal exception during runner processing."""

    async def process(self, world: World) -> None:
        """Raise a test exception."""
        _ = world
        raise RuntimeError("runner failed")


class HangingSystem:
    """System that waits until the runner task is externally cancelled."""

    async def process(self, world: World) -> None:
        """Block forever until cancellation propagates into the system."""
        _ = world
        await asyncio.Event().wait()


class TwoUserTurnSystem:
    """System that emits two user-turn LLM chains within one runner run."""

    def __init__(self) -> None:
        self._tick = 0
        self.entity_id: int | None = None

    async def process(self, world: World) -> None:
        """Emit a user input and LLM generation per tick, then terminate."""
        if self.entity_id is None:
            self.entity_id = int(world.create_entity())
        entity_id = self.entity_id
        user_text = "first question" if self._tick == 0 else "second question"
        answer_text = "first answer" if self._tick == 0 else "second answer"
        await world.event_bus.publish(
            UserInputReceivedEvent(
                entity_id=entity_id,
                prompt="You> ",
                text=user_text,
            )
        )
        await world.event_bus.publish(
            LLMObservationStartedEvent(
                entity_id=entity_id,
                provider_id="fake-provider",
                model=f"turn-model-{self._tick}",
                operation="reasoning",
                messages=[Message(role="user", content=user_text)],
            )
        )
        await world.event_bus.publish(
            LLMObservationCompletedEvent(
                entity_id=entity_id,
                provider_id="fake-provider",
                model=f"turn-model-{self._tick}",
                operation="reasoning",
                messages=[Message(role="user", content=user_text)],
                response_message=Message(role="assistant", content=answer_text),
                usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            )
        )
        self._tick += 1
        if self._tick >= 2:
            world.add_component(world.create_entity(), TerminalComponent(reason="done"))


class UserTurnSubagentSystem:
    """System that launches a real subagent inside one interactive user turn."""

    def __init__(self, entity_id: int) -> None:
        self.entity_id = entity_id
        self._ran = False

    async def process(self, world: World) -> None:
        """Publish user input, call the installed subagent tool, then terminate."""
        if self._ran:
            return
        self._ran = True
        await world.event_bus.publish(
            UserInputReceivedEvent(
                entity_id=self.entity_id,
                prompt="You> ",
                text="delegate inside this turn",
            )
        )
        registry = world.get_component(self.entity_id, ToolRegistryComponent)
        assert registry is not None
        handler = registry.handlers["subagent"]
        await handler(category="worker", prompt="Do child task")
        world.add_component(world.create_entity(), TerminalComponent(reason="done"))


class UserTurnBackgroundSubagentSystem:
    """System that queues a background subagent inside one interactive turn."""

    def __init__(self, entity_id: int) -> None:
        self.entity_id = entity_id
        self._ran = False
        self.session_id: str | None = None

    async def process(self, world: World) -> None:
        """Publish user input, queue a background subagent, then terminate."""
        if self._ran:
            return
        self._ran = True
        await world.event_bus.publish(
            UserInputReceivedEvent(
                entity_id=self.entity_id,
                prompt="You> ",
                text="queue delayed delegation inside this turn",
            )
        )
        registry = world.get_component(self.entity_id, ToolRegistryComponent)
        assert registry is not None
        handler = registry.handlers["subagent"]
        result = await handler(
            category="worker",
            prompt="Do delayed child task",
            background=True,
        )
        payload = json.loads(result)
        self.session_id = payload["session_id"]
        world.add_component(world.create_entity(), TerminalComponent(reason="done"))


class TwoTurnBackgroundSubagentSystem:
    """System that launches a background subagent in turn one, then starts turn two."""

    def __init__(self, entity_id: int) -> None:
        self.entity_id = entity_id
        self._tick = 0
        self.session_id: str | None = None

    async def process(self, world: World) -> None:
        """Exercise background completion while a later user-turn trace is active."""
        registry = world.get_component(self.entity_id, ToolRegistryComponent)
        assert registry is not None
        handler = registry.handlers["subagent"]
        if self._tick == 0:
            await world.event_bus.publish(
                UserInputReceivedEvent(
                    entity_id=self.entity_id,
                    prompt="You> ",
                    text="turn one launches background",
                )
            )
            result = await handler(
                category="worker",
                prompt="Complete during later turn",
                background=True,
            )
            payload = json.loads(result)
            self.session_id = payload["session_id"]
        elif self._tick == 1:
            await world.event_bus.publish(
                UserInputReceivedEvent(
                    entity_id=self.entity_id,
                    prompt="You> ",
                    text="turn two is active",
                )
            )
            await asyncio.sleep(0.05)
            world.add_component(world.create_entity(), TerminalComponent(reason="done"))
        self._tick += 1


class DelayedFakeModel(FakeModel):
    """Fake model that waits before returning a response."""

    def __init__(self, responses: list[CompletionResult], model_id: str) -> None:
        super().__init__(responses=responses, model_id=model_id)
        self.release = asyncio.Event()

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        stream: bool = False,
        response_format: dict[str, object] | None = None,
    ) -> CompletionResult:
        """Wait until released, then return the next fake response."""
        del tools, stream, response_format
        _ = messages
        await self.release.wait()
        result = await super().complete(messages)
        if not isinstance(result, CompletionResult):
            raise RuntimeError("expected completion result")
        return result


class FailingSink(RecordingTelemetrySink):
    """Recording sink that raises on telemetry emission."""

    async def emit(self, record: object) -> None:
        """Raise to prove EventBus subscriber isolation preserves the run."""
        _ = record
        raise RuntimeError("sink failed")


class ScoreFailingSink(RecordingTelemetrySink):
    """Recording sink that raises when a score is emitted."""

    async def score(self, score: object) -> None:
        """Raise to exercise completion cleanup when scoring fails."""
        _ = score
        raise RuntimeError("score failed")


def _unique_turn_traces(sink: RecordingTelemetrySink) -> list[TelemetryRecord]:
    """Return each user-turn trace once while preserving first-emission order."""
    traces_by_id = {
        record.observation_id: record
        for record in sink.records
        if record.kind == "trace" and record.name == "user.turn"
    }
    return list(traces_by_id.values())


@pytest.mark.asyncio
async def test_runner_success_creates_one_trace() -> None:
    """A successful runner run creates one closed trace without empty tick spans."""
    world = World()
    world.register_system(TerminatingSystem(), priority=0)
    sink = RecordingTelemetrySink()
    install_observability(world, sink)

    await Runner().run(world, max_ticks=5)

    trace_records = [record for record in sink.records if record.kind == "trace"]
    assert len(trace_records) == 1
    trace = trace_records[0]
    assert trace.name == "runner.run"
    assert trace.status == "success"
    assert trace.end_time is not None
    assert trace.latency_ms is not None
    assert trace.metadata == {
        "max_ticks": 5,
        "start_tick": 0,
        "active_entities_start": 0,
        "active_entities_end": 2,
        "reason": "terminal_component",
        "ticks": 1,
    }

    assert {record.run_id for record in sink.records} == {trace.run_id}
    assert {record.trace_id for record in sink.records} == {trace.trace_id}
    assert not any(record.name == "runner.tick" for record in sink.records)
    assert any(record.name.endswith("TerminatingSystem") for record in sink.records)
    assert current_trace_id() is None
    assert current_run_id() is None


@pytest.mark.asyncio
async def test_empty_runner_and_noisy_system_lifecycle_spans_are_suppressed() -> None:
    """Empty runner tick plus Reasoning/Subagent lifecycle records are not traced."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    token = set_run_context(trace_id="trace-empty", run_id="run-empty")
    noisy_systems = {
        "ecs_agent.systems.error_handling.ErrorHandlingSystem",
        "ecs_agent.systems.reasoning.ReasoningSystem",
        "ecs_agent.systems.subagent.SubagentSystem",
        "ecs_agent.systems.user_input.UserInputSystem",
        "ecs_agent.systems.tool_execution.ToolExecutionSystem",
        "ecs_agent.systems.terminal_cleanup.TerminalCleanupSystem",
    }

    try:
        await world.event_bus.publish(
            RunStartedEvent(max_ticks=1, start_tick=0, active_entities=0)
        )
        await world.event_bus.publish(
            RunnerTickStartedEvent(tick=0, active_entities=0)
        )
        await world.event_bus.publish(
            RunnerTickCompletedEvent(
                tick=0,
                status="success",
                duration_seconds=0.001,
                active_entities=0,
            )
        )
        for system_name in noisy_systems:
            await world.event_bus.publish(SystemExecutionStartedEvent(system=system_name))
            await world.event_bus.publish(
                SystemExecutionCompletedEvent(
                    system=system_name,
                    status="success",
                    duration_seconds=0.001,
                )
            )
        await world.event_bus.publish(
            RunCompletedEvent(
                status="success",
                reason="manual",
                duration_seconds=0.01,
                ticks=1,
                active_entities=0,
            )
        )
    finally:
        reset_run_context(token)

    assert not any(record.name == "runner.tick" for record in sink.records)
    assert not any(record.name in noisy_systems for record in sink.records)


@pytest.mark.asyncio
async def test_runner_max_ticks_records_score_and_closes_trace() -> None:
    """A max_ticks run closes its trace and emits required summary scores."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)

    await Runner().run(world, max_ticks=0)

    trace_records = [record for record in sink.records if record.kind == "trace"]
    assert len(trace_records) == 1
    trace = trace_records[0]
    assert trace.status == "success"
    assert trace.metadata is not None
    assert trace.metadata["reason"] == "max_ticks"

    scores = {score.name: score.value for score in sink.scores}
    assert scores == {
        "agent_tick_count": 0,
        "agent_latency_ms": pytest.approx(trace.latency_ms),
        "agent_error_count": 0,
        "estimated_context_pressure": 0.0,
        "max_ticks_reached": True,
    }
    assert {score.observation_id for score in sink.scores} == {trace.observation_id}


@pytest.mark.asyncio
async def test_runner_exception_creates_one_error_trace_and_cleans_state() -> None:
    """A runner exception closes one error trace and clears subscriber state."""
    world = World()
    world.register_system(RaisingSystem(), priority=0)
    sink = RecordingTelemetrySink()
    install_observability(world, sink)

    with pytest.raises(ExceptionGroup):
        await Runner().run(world, max_ticks=5)

    trace_records = [record for record in sink.records if record.kind == "trace"]
    assert len(trace_records) == 1
    trace = trace_records[0]
    assert trace.status == "error"
    assert trace.metadata is not None
    assert trace.metadata["reason"] == "exception"
    assert trace.end_time is not None
    scores = {score.name: score.value for score in sink.scores}
    assert scores["agent_tick_count"] == 1
    assert scores["agent_latency_ms"] == pytest.approx(trace.latency_ms)
    assert scores["agent_error_count"] == 0
    assert scores["max_ticks_reached"] is False

    subscriber = getattr(world, "_ecs_agent_observability_subscriber")
    assert subscriber.trace_states == {}


@pytest.mark.asyncio
async def test_runner_external_cancellation_creates_one_cancelled_trace() -> None:
    """External cancellation propagates while closing one cancelled trace."""
    world = World()
    world.register_system(HangingSystem(), priority=0)
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    started = asyncio.Event()

    async def on_started(event: RunStartedEvent) -> None:
        _ = event
        started.set()

    world.event_bus.subscribe(RunStartedEvent, on_started)
    run_task = asyncio.create_task(Runner().run(world, max_ticks=5))
    await started.wait()

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    trace_records = [record for record in sink.records if record.kind == "trace"]
    assert len(trace_records) == 1
    trace = trace_records[0]
    assert trace.status == "cancelled"
    assert trace.metadata is not None
    assert trace.metadata["reason"] == "external_cancellation"
    assert trace.end_time is not None
    scores = {score.name: score.value for score in sink.scores}
    assert scores["agent_tick_count"] == 0
    assert scores["agent_latency_ms"] == pytest.approx(trace.latency_ms)
    assert scores["agent_error_count"] == 0
    assert scores["max_ticks_reached"] is False

    subscriber = getattr(world, "_ecs_agent_observability_subscriber")
    assert subscriber.trace_states == {}


@pytest.mark.asyncio
async def test_subscriber_keeps_separate_trace_state_per_run() -> None:
    """Sequential runs on one world produce separate traces and clean state."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)

    await Runner().run(world, max_ticks=0)
    await Runner().run(world, max_ticks=0)

    traces = [record for record in sink.records if record.kind == "trace"]
    assert len(traces) == 2
    assert len({trace.run_id for trace in traces}) == 2
    assert len({trace.trace_id for trace in traces}) == 2
    subscriber = getattr(world, "_ecs_agent_observability_subscriber")
    assert subscriber.trace_states == {}


@pytest.mark.asyncio
async def test_user_inputs_create_separate_turn_traces_with_child_generations() -> None:
    """One long runner run creates one trace per user input turn."""
    world = World()
    system = TwoUserTurnSystem()
    world.register_system(system, priority=0)
    sink = RecordingTelemetrySink()
    install_observability(world, sink)

    await Runner().run(world, max_ticks=5)

    turn_traces = _unique_turn_traces(sink)
    assert len(turn_traces) == 2
    assert [trace.input for trace in turn_traces] == [
        {"text": "first question"},
        {"text": "second question"},
    ]
    assert len({trace.trace_id for trace in turn_traces}) == 2

    generations = [record for record in sink.records if record.kind == "generation"]
    assert len(generations) == 2
    assert [generation.trace_id for generation in generations] == [
        trace.trace_id for trace in turn_traces
    ]
    assert [generation.parent_observation_id for generation in generations] == [
        trace.observation_id for trace in turn_traces
    ]


@pytest.mark.asyncio
async def test_user_turn_root_is_emitted_before_workflow_children() -> None:
    """Interactive turn roots are visible before child observations are emitted."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    token = set_run_context(trace_id="run-trace", run_id="run-one")

    try:
        await world.event_bus.publish(
            RunStartedEvent(max_ticks=1, start_tick=0, active_entities=0)
        )
        entity_id = world.create_entity()
        await world.event_bus.publish(
            UserInputReceivedEvent(
                entity_id=entity_id,
                prompt="You> ",
                text="draft the plan",
            )
        )
        await world.event_bus.publish(
            WorkflowStateEvaluatedEvent(
                entity_id=entity_id,
                workflow_id="plan_flow",
                state_id="draft",
                current_state_id="review",
                tick=0,
                matched_transition_ids=["draft_to_review"],
                committed_transition_id="draft_to_review",
                from_state_id="draft",
                to_state_id="review",
                transition_history=["draft_to_review"],
                status="transition",
            )
        )
        await world.event_bus.publish(
            RunCompletedEvent(
                status="success",
                reason="manual",
                duration_seconds=0.01,
                ticks=1,
                active_entities=1,
            )
        )
    finally:
        reset_run_context(token)

    user_turn = next(record for record in sink.records if record.name == "user.turn")
    workflow_state = next(record for record in sink.records if record.name == "workflow.state")

    assert sink.records.index(user_turn) < sink.records.index(workflow_state)
    assert workflow_state.trace_id == user_turn.trace_id
    assert workflow_state.parent_observation_id == user_turn.observation_id


@pytest.mark.asyncio
async def test_pre_turn_workflow_state_is_reparented_to_user_turn() -> None:
    """Workflow telemetry before input waits for the interactive turn root."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    token = set_run_context(trace_id="run-trace", run_id="run-one")

    try:
        await world.event_bus.publish(
            RunStartedEvent(max_ticks=1, start_tick=0, active_entities=0)
        )
        entity_id = world.create_entity()
        await world.event_bus.publish(
            WorkflowStateEvaluatedEvent(
                entity_id=entity_id,
                workflow_id="plan_flow",
                state_id="draft",
                current_state_id="draft",
                tick=0,
                matched_transition_ids=[],
                transition_history=[],
                status="no_match",
            )
        )
        assert not any(record.name == "workflow.state" for record in sink.records)

        await world.event_bus.publish(
            UserInputReceivedEvent(
                entity_id=entity_id,
                prompt="You> ",
                text="continue the plan",
            )
        )
    finally:
        reset_run_context(token)

    user_turn = next(record for record in sink.records if record.name == "user.turn")
    workflow_state = next(record for record in sink.records if record.name == "workflow.state")

    assert sink.records.index(user_turn) < sink.records.index(workflow_state)
    assert workflow_state.trace_id == user_turn.trace_id
    assert workflow_state.parent_observation_id == user_turn.observation_id


@pytest.mark.asyncio
async def test_subagent_child_generation_uses_active_user_turn_trace() -> None:
    """Subagent child-world telemetry stays under the active user-turn trace."""
    world = World()
    parent_model = FakeModel(responses=[], model_id="unused-parent")
    child_model = FakeModel(
        responses=["child answer"],
        model_id="child-model",
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=parent_model))
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(entity_id, SubagentSessionTableComponent(sessions={}))
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=child_model,
                    system_prompt="You are a worker.",
                    max_ticks=2,
                )
            }
        ),
    )
    world.register_system(SubagentSystem(priority=-1), priority=-1)
    world.register_system(UserTurnSubagentSystem(int(entity_id)), priority=0)
    sink = RecordingTelemetrySink()
    install_observability(world, sink)

    await Runner().run(world, max_ticks=5)

    turn_trace = next(
        record
        for record in sink.records
        if record.kind == "trace" and record.name == "user.turn"
    )
    subagent_span = next(record for record in sink.records if record.name == "subagent.worker")
    child_generation = next(record for record in sink.records if record.model == "child-model")

    assert subagent_span.trace_id == turn_trace.trace_id
    assert subagent_span.parent_observation_id == turn_trace.observation_id
    assert child_generation.trace_id == turn_trace.trace_id
    assert child_generation.parent_observation_id == subagent_span.observation_id


@pytest.mark.asyncio
async def test_background_subagent_session_captures_launch_user_turn_trace() -> None:
    """Queued background subagents persist the launch user-turn trace context."""
    world = World()
    parent_model = FakeModel(responses=[], model_id="unused-parent")
    child_model = FakeModel(responses=["child answer"], model_id="child-model")
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=parent_model))
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(entity_id, SubagentSessionTableComponent(sessions={}))
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=child_model,
                    system_prompt="You are a worker.",
                    max_ticks=2,
                )
            }
        ),
    )
    world.register_system(SubagentSystem(priority=-1, max_background_concurrency=1), priority=-1)
    system = UserTurnBackgroundSubagentSystem(int(entity_id))
    world.register_system(system, priority=0)
    sink = RecordingTelemetrySink()
    install_observability(world, sink)

    await Runner().run(world, max_ticks=5)

    assert system.session_id is not None
    turn_trace = next(
        record
        for record in sink.records
        if record.kind == "trace" and record.name == "user.turn"
    )
    table = world.get_component(entity_id, SubagentSessionTableComponent)
    assert table is not None
    session = table.sessions[system.session_id]
    assert session.launch_trace_id == turn_trace.trace_id
    assert session.launch_run_id == turn_trace.run_id
    assert session.launch_parent_observation_id == turn_trace.observation_id

    for _ in range(50):
        has_subagent_span = any(
            record.name == "subagent.worker" for record in sink.records
        )
        has_child_generation = any(record.model == "child-model" for record in sink.records)
        if has_subagent_span and has_child_generation:
            break
        await asyncio.sleep(0.01)

    subagent_span = next(record for record in sink.records if record.name == "subagent.worker")
    child_generation = next(record for record in sink.records if record.model == "child-model")
    assert subagent_span.trace_id == turn_trace.trace_id
    assert subagent_span.parent_observation_id == turn_trace.observation_id
    assert child_generation.trace_id == turn_trace.trace_id
    assert child_generation.parent_observation_id == subagent_span.observation_id


@pytest.mark.asyncio
async def test_background_subagent_completion_uses_launch_turn_when_later_turn_active() -> None:
    """Background completion stays on its launch turn even if another turn is active."""
    world = World()
    parent_model = FakeModel(responses=[], model_id="unused-parent")
    child_model = DelayedFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="child answer"))],
        model_id="child-model",
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=parent_model))
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(entity_id, SubagentSessionTableComponent(sessions={}))
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=child_model,
                    system_prompt="You are a worker.",
                    max_ticks=2,
                )
            }
        ),
    )
    world.register_system(SubagentSystem(priority=-1, max_background_concurrency=1), priority=-1)
    system = TwoTurnBackgroundSubagentSystem(int(entity_id))
    world.register_system(system, priority=0)
    sink = RecordingTelemetrySink()
    install_observability(world, sink)

    async def release_after_second_turn(event: UserInputReceivedEvent) -> None:
        if event.text == "turn two is active":
            child_model.release.set()

    world.event_bus.subscribe(UserInputReceivedEvent, release_after_second_turn)

    await Runner().run(world, max_ticks=5)

    turn_traces = _unique_turn_traces(sink)
    assert [trace.input for trace in turn_traces] == [
        {"text": "turn one launches background"},
        {"text": "turn two is active"},
    ]
    first_turn, second_turn = turn_traces
    subagent_spans = [record for record in sink.records if record.name == "subagent.worker"]
    assert len(subagent_spans) == 1
    subagent_span = subagent_spans[0]
    child_generation = next(record for record in sink.records if record.model == "child-model")
    assert subagent_span.trace_id == first_turn.trace_id
    assert subagent_span.trace_id != second_turn.trace_id
    assert subagent_span.parent_observation_id == first_turn.observation_id
    assert child_generation.trace_id == first_turn.trace_id
    assert child_generation.parent_observation_id == subagent_span.observation_id


@pytest.mark.asyncio
async def test_error_events_increment_completed_trace_score() -> None:
    """Error events are linked to the active run and counted on completion."""
    world = World()
    system = ErrorEventSystem()
    world.register_system(system, priority=0)
    sink = RecordingTelemetrySink()
    install_observability(world, sink)

    await Runner().run(world, max_ticks=5)

    error_records = [record for record in sink.records if record.name == "error.occurred"]
    assert len(error_records) == 1
    assert error_records[0].status == "error"
    assert error_records[0].entity_id == system.entity_id
    assert error_records[0].error == "recoverable failure"

    error_scores = [score for score in sink.scores if score.name == "agent_error_count"]
    assert len(error_scores) == 1
    assert error_scores[0].value == 1


@pytest.mark.asyncio
async def test_subscriber_exceptions_are_isolated_by_event_bus() -> None:
    """Sink failures inside the subscriber do not fail the runner."""
    world = World()
    world.register_system(TerminatingSystem(), priority=0)
    install_observability(world, FailingSink())
    completed: list[bool] = []

    async def on_started(event: RunStartedEvent) -> None:
        _ = event
        completed.append(True)

    world.event_bus.subscribe(RunStartedEvent, on_started)

    await Runner().run(world, max_ticks=5)

    assert completed == [True]


@pytest.mark.asyncio
async def test_run_completed_cleans_trace_state_when_score_emission_fails() -> None:
    """Score emission failures still clean up trace state for later runs."""
    world = World()
    world.register_system(TerminatingSystem(), priority=0)
    sink = ScoreFailingSink()
    install_observability(world, sink)

    await Runner().run(world, max_ticks=5)

    subscriber = getattr(world, "_ecs_agent_observability_subscriber")
    assert subscriber.trace_states == {}

    await Runner().run(world, max_ticks=5)
    assert subscriber.trace_states == {}
