import asyncio
import time
from dataclasses import dataclass

import pytest

from ecs_agent.core.system import SystemExecutor
from ecs_agent.core.world import World


@dataclass(slots=True)
class LoggingSystem:
    name: str
    log: list[str]

    async def process(self, world: World) -> None:
        _ = world
        self.log.append(self.name)


@dataclass(slots=True)
class SlowSystem:
    name: str
    delay: float
    log: list[str]

    async def process(self, world: World) -> None:
        _ = world
        await asyncio.sleep(self.delay)
        self.log.append(self.name)


@pytest.mark.asyncio
async def test_system_executor_register_and_execute_single_system() -> None:
    executor = SystemExecutor()
    world = World()
    log: list[str] = []
    executor.register(LoggingSystem(name="single", log=log), priority=0)

    await executor.execute(world)
    assert log == ["single"]


@pytest.mark.asyncio
async def test_system_executor_executes_lower_priority_first() -> None:
    executor = SystemExecutor()
    world = World()
    log: list[str] = []

    executor.register(LoggingSystem(name="p1", log=log), priority=1)
    executor.register(LoggingSystem(name="p0", log=log), priority=0)

    await executor.execute(world)
    assert log == ["p0", "p1"]


@pytest.mark.asyncio
async def test_system_executor_runs_same_priority_in_parallel() -> None:
    executor = SystemExecutor()
    world = World()
    log: list[str] = []

    executor.register(SlowSystem(name="a", delay=0.1, log=log), priority=0)
    executor.register(SlowSystem(name="b", delay=0.1, log=log), priority=0)

    start = time.monotonic()
    await executor.execute(world)
    elapsed = time.monotonic() - start

    assert set(log) == {"a", "b"}
    assert elapsed < 0.15


@pytest.mark.asyncio
async def test_system_protocol_structural_typing_works_with_world_registration() -> (
    None
):
    world = World()
    log: list[str] = []
    world.register_system(LoggingSystem(name="typed", log=log), priority=0)

    await world.process()
    assert log == ["typed"]
