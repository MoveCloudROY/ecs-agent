"""Observability installation handle and runner context tests."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import Runner, World
from ecs_agent.observability import (
    RecordingTelemetrySink,
    current_observation_stack,
    current_run_id,
    current_trace_id,
    push_observation,
    reset_observation,
)
from ecs_agent.types import (
    RunCompletedEvent,
    RunStartedEvent,
    RunnerTickCompletedEvent,
    RunnerTickStartedEvent,
)


class FlushShutdownSink(RecordingTelemetrySink):
    """Recording sink that counts lifecycle method calls."""

    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0
        self.shutdown_count = 0

    async def flush(self) -> None:
        """Record flush calls."""
        self.flush_count += 1

    async def shutdown(self) -> None:
        """Record shutdown calls."""
        self.shutdown_count += 1


class TerminatingSystem:
    """System that terminates a run on its first tick."""

    async def process(self, world: World) -> None:
        """Attach a terminal component."""
        world.add_component(world.create_entity(), TerminalComponent(reason="done"))


class ErrorSystem:
    """System that raises during processing."""

    async def process(self, world: World) -> None:
        """Raise a test error."""
        _ = world
        raise RuntimeError("runner failed")


class HangingSystem:
    """System that waits until its runner task is cancelled."""

    async def process(self, world: World) -> None:
        """Wait forever for external cancellation."""
        _ = world
        await asyncio.Event().wait()


def test_observability_public_surface_importable() -> None:
    """Install API is importable from the observability package."""
    from ecs_agent.observability import (
        ObservabilityHandle,
        install_observability,
        uninstall_observability,
    )

    assert callable(install_observability)
    assert callable(uninstall_observability)
    assert ObservabilityHandle.__name__ == "ObservabilityHandle"


@pytest.mark.asyncio
async def test_observability_handle_delegates_flush_shutdown_and_uninstall() -> None:
    """Handle methods delegate sink lifecycle and remove world bookkeeping."""
    from ecs_agent.observability import install_observability, uninstall_observability

    world = World()
    sink = FlushShutdownSink()
    handle = install_observability(world, sink, config={"enabled": True})

    await handle.flush()
    await handle.shutdown()
    removed = handle.uninstall()

    assert removed is handle
    assert sink.flush_count == 1
    assert sink.shutdown_count == 1
    assert uninstall_observability(world) is None


def test_install_observability_is_idempotent_for_same_sink() -> None:
    """Installing the same sink on a world returns the existing handle."""
    from ecs_agent.observability import install_observability

    world = World()
    sink = RecordingTelemetrySink()

    first = install_observability(world, sink)
    second = install_observability(world, sink)

    assert second is first


def test_install_observability_rejects_different_sink() -> None:
    """Installing a different sink on an observed world fails clearly."""
    from ecs_agent.observability import install_observability

    world = World()
    install_observability(world, RecordingTelemetrySink())

    with pytest.raises(ValueError, match="observability is already installed"):
        install_observability(world, RecordingTelemetrySink())


def test_run_context_scopes_observation_stack() -> None:
    """Run context clears observation stack during the run and restores it after reset."""
    from ecs_agent.observability import reset_run_context, set_run_context

    outer_token = push_observation("outer")
    run_token = set_run_context(trace_id="trace", run_id="run")

    assert current_observation_stack() == ()
    push_observation("inner")
    assert current_observation_stack() == ("inner",)

    reset_run_context(run_token)
    assert current_observation_stack() == ("outer",)

    reset_observation(outer_token)


@pytest.mark.asyncio
async def test_uninstall_observability_unsubscribes_registered_handlers() -> None:
    """Uninstall removes every observability subscription and is idempotent."""
    from ecs_agent.observability import install_observability, uninstall_observability

    world = World()
    sink = RecordingTelemetrySink()
    handle = install_observability(world, sink)
    seen: list[int] = []

    async def on_started(event: RunStartedEvent) -> None:
        seen.append(event.active_entities)

    world.event_bus.subscribe(RunStartedEvent, on_started)
    subscriptions = getattr(world, "_ecs_agent_observability_subscriptions")
    subscriptions.append((RunStartedEvent, on_started))

    assert handle.uninstall() is handle
    assert handle.uninstall() is None
    assert uninstall_observability(world) is None

    await world.event_bus.publish(RunStartedEvent(max_ticks=1, start_tick=0, active_entities=3))
    assert seen == []


@pytest.mark.asyncio
async def test_runner_publish_events_share_run_context_and_reset_after_success() -> None:
    """Runner lifecycle publishes share one run context and reset afterwards."""
    world = World()
    world.register_system(TerminatingSystem(), priority=0)
    captured: list[tuple[str, str | None, str | None]] = []

    async def capture(event: Any) -> None:
        captured.append((type(event).__name__, current_trace_id(), current_run_id()))

    world.event_bus.subscribe(RunStartedEvent, capture)
    world.event_bus.subscribe(RunnerTickStartedEvent, capture)
    world.event_bus.subscribe(RunnerTickCompletedEvent, capture)
    world.event_bus.subscribe(RunCompletedEvent, capture)

    await Runner().run(world, max_ticks=5)

    context_pairs = {(trace_id, run_id) for _, trace_id, run_id in captured}
    assert [name for name, _, _ in captured] == [
        "RunStartedEvent",
        "RunnerTickStartedEvent",
        "RunnerTickCompletedEvent",
        "RunCompletedEvent",
    ]
    assert len(context_pairs) == 1
    trace_id, run_id = next(iter(context_pairs))
    assert trace_id is not None
    assert run_id is not None
    assert current_trace_id() is None
    assert current_run_id() is None


@pytest.mark.asyncio
async def test_run_context_resets_after_runner_error() -> None:
    """Runner resets context when world processing raises."""
    world = World()
    world.register_system(ErrorSystem(), priority=0)
    completed_contexts: list[tuple[str | None, str | None]] = []

    async def on_completed(event: RunCompletedEvent) -> None:
        _ = event
        completed_contexts.append((current_trace_id(), current_run_id()))

    world.event_bus.subscribe(RunCompletedEvent, on_completed)

    with pytest.raises(ExceptionGroup):
        await Runner().run(world, max_ticks=5)

    assert len(completed_contexts) == 1
    assert completed_contexts[0][0] is not None
    assert completed_contexts[0][1] is not None
    assert current_trace_id() is None
    assert current_run_id() is None


@pytest.mark.asyncio
async def test_run_context_resets_after_max_ticks() -> None:
    """Runner resets context after max_ticks completion."""
    world = World()
    completed_contexts: list[tuple[str | None, str | None]] = []

    async def on_completed(event: RunCompletedEvent) -> None:
        _ = event
        completed_contexts.append((current_trace_id(), current_run_id()))

    world.event_bus.subscribe(RunCompletedEvent, on_completed)

    await Runner().run(world, max_ticks=0)

    assert len(completed_contexts) == 1
    assert completed_contexts[0][0] is not None
    assert completed_contexts[0][1] is not None
    assert current_trace_id() is None
    assert current_run_id() is None


@pytest.mark.asyncio
async def test_run_context_resets_after_external_cancellation() -> None:
    """Runner resets context when external cancellation propagates."""
    world = World()
    world.register_system(HangingSystem(), priority=0)
    started = asyncio.Event()
    started_contexts: list[tuple[str | None, str | None]] = []

    async def on_started(event: RunStartedEvent) -> None:
        _ = event
        started_contexts.append((current_trace_id(), current_run_id()))
        started.set()

    world.event_bus.subscribe(RunStartedEvent, on_started)
    run_task = asyncio.create_task(Runner().run(world, max_ticks=5))
    await started.wait()

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert started_contexts[0][0] is not None
    assert started_contexts[0][1] is not None
    assert current_trace_id() is None
    assert current_run_id() is None
