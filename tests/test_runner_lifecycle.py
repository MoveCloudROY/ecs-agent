"""Behavior contracts for repeatable and exclusively driven runs."""

import asyncio

import pytest

from ecs_agent.components import InterruptionComponent, RunnerStateComponent, TerminalComponent
from ecs_agent.core import Runner, World
from ecs_agent.core.run_result import RunStatus
from ecs_agent.types import InterruptionReason, RunnerTickStartedEvent


class Counter:
    """Record actual system executions."""

    def __init__(self) -> None:
        self.count = 0

    async def process(self, world: World) -> None:
        self.count += 1


async def test_repeated_runs_keep_their_budget_and_cumulative_tick() -> None:
    world = World()
    counter = Counter()
    world.register_system(counter, priority=0)
    ticks: list[int] = []

    async def observe(event: RunnerTickStartedEvent) -> None:
        ticks.append(event.tick)

    world.event_bus.subscribe(RunnerTickStartedEvent, observe)
    first = await Runner().run(world, max_ticks=1)
    second = await Runner().run(world, max_ticks=5)
    assert first.status is second.status is RunStatus.MAX_TICKS
    assert (first.ticks, second.ticks, counter.count) == (1, 5, 6)
    assert ticks == list(range(6))
    assert not list(world.query(TerminalComponent))
    assert list(world.query(RunnerStateComponent))[0][1][0].current_tick == 6


@pytest.mark.parametrize("interrupted", [False, True])
async def test_preexisting_stop_executes_no_system(interrupted: bool) -> None:
    world = World()
    entity = world.create_entity()
    if interrupted:
        world.add_component(entity, InterruptionComponent(reason=InterruptionReason.USER_REQUESTED))
    else:
        world.add_component(entity, TerminalComponent(reason="test_complete"))
    counter = Counter()
    world.register_system(counter, priority=0)
    result = await Runner().run(world, max_ticks=10, target_entity=entity)
    assert counter.count == result.ticks == 0
    assert result.reason == ("interruption_component" if interrupted else "test_complete")


async def test_legacy_absolute_budget_is_unchanged() -> None:
    world = World()
    counter = Counter()
    world.register_system(counter, priority=0)
    result = await Runner().run(world, start_tick=5, max_ticks=8)
    assert result.ticks == counter.count == 3
    assert list(world.query(RunnerStateComponent))[0][1][0].current_tick == 8


async def test_concurrent_runners_are_rejected_and_cancel_releases_world() -> None:
    world = World()
    entered = asyncio.Event()

    class Waiting:
        async def process(self, world: World) -> None:
            entered.set()
            await asyncio.Event().wait()

    handle = world.register_system(Waiting(), priority=0)
    runner = Runner()
    task = asyncio.create_task(runner.run(world))
    await asyncio.wait_for(entered.wait(), timeout=2)
    try:
        with pytest.raises(RuntimeError, match="already driving"):
            await Runner().run(world)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert runner.last_result is not None
    assert runner.last_result.status is RunStatus.CANCELLED
    world.remove_system(handle)
    result = await runner.run(world, max_ticks=1)
    assert result.ticks == 1


async def test_zero_budget_does_not_execute_and_negative_budget_is_rejected() -> None:
    world = World()
    counter = Counter()
    world.register_system(counter, priority=0)
    assert (await Runner().run(world, max_ticks=0)).ticks == 0
    with pytest.raises(ValueError, match="non-negative"):
        await Runner().run(world, max_ticks=-1)
    assert counter.count == 0
