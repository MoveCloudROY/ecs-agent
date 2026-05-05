"""Observability EventBus subscriber trace lifecycle tests."""

from __future__ import annotations

import asyncio

import pytest

from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import Runner, World
from ecs_agent.observability import (
    RecordingTelemetrySink,
    current_run_id,
    current_trace_id,
    install_observability,
    reset_run_context,
    set_run_context,
)
from ecs_agent.types import (
    ErrorOccurredEvent,
    RunCompletedEvent,
    RunnerTickCompletedEvent,
    RunnerTickStartedEvent,
    RunStartedEvent,
    SystemExecutionCompletedEvent,
    SystemExecutionStartedEvent,
)


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
