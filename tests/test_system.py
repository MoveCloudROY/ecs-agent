import asyncio
import time
from dataclasses import dataclass

import pytest

from ecs_agent.core.system import SystemExecutor
from ecs_agent.core.world import World
from ecs_agent.types import SystemHandle


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


class TestSystemLogging:
    """Test System execution logging."""

    @pytest.mark.asyncio
    async def test_system_executor_emits_system_start_event(self, capsys) -> None:
        """Test that system executor emits system_start for each system."""
        import json
        import sys
        from ecs_agent.logging import configure_logging, STANDARD_EVENT_NAMES

        configure_logging(json_output=True, level="INFO")

        # Force reimport to pick up new logger config
        for mod in ["ecs_agent.core.system", "ecs_agent.core.world"]:
            sys.modules.pop(mod, None)

        from ecs_agent.core.system import SystemExecutor
        from ecs_agent.core.world import World

        executor = SystemExecutor()
        world = World()
        log: list[str] = []
        executor.register(LoggingSystem(name="test_system", log=log), priority=0)

        await executor.execute(world)

        captured = capsys.readouterr()
        # Parse only JSON lines
        events = []
        for line in captured.out.strip().split("\n"):
            if line.strip() and line.strip().startswith("{"):
                events.append(json.loads(line))
        system_start_events = [
            e for e in events if e.get("event") == STANDARD_EVENT_NAMES["SYSTEM_START"]
        ]

        assert len(system_start_events) >= 1
        event = system_start_events[0]
        assert "system" in event
        assert "LoggingSystem" in event["system"]

    @pytest.mark.asyncio
    async def test_system_executor_emits_system_complete_event(self, capsys) -> None:
        """Test that system executor emits system_complete with duration."""
        import json
        import sys
        from ecs_agent.logging import configure_logging, STANDARD_EVENT_NAMES

        configure_logging(json_output=True, level="INFO")

        # Force reimport to pick up new logger config
        for mod in ["ecs_agent.core.system", "ecs_agent.core.world"]:
            sys.modules.pop(mod, None)

        from ecs_agent.core.system import SystemExecutor
        from ecs_agent.core.world import World

        executor = SystemExecutor()
        world = World()
        log: list[str] = []
        executor.register(LoggingSystem(name="test_system", log=log), priority=0)

        await executor.execute(world)

        captured = capsys.readouterr()
        # Parse only JSON lines
        events = []
        for line in captured.out.strip().split("\n"):
            if line.strip() and line.strip().startswith("{"):
                events.append(json.loads(line))
        system_complete_events = [
            e
            for e in events
            if e.get("event") == STANDARD_EVENT_NAMES["SYSTEM_COMPLETE"]
        ]

        assert len(system_complete_events) >= 1
        event = system_complete_events[0]
        assert "system" in event
        assert "duration_ms" in event
        assert isinstance(event["duration_ms"], (int, float))
        assert event["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_system_executor_emits_system_error_on_exception(
        self, capsys
    ) -> None:
        """Test that system executor emits system_error when system raises exception."""
        import json
        import sys
        from ecs_agent.logging import configure_logging, STANDARD_EVENT_NAMES

        configure_logging(json_output=True, level="ERROR")

        # Force reimport to pick up new logger config
        for mod in ["ecs_agent.core.system", "ecs_agent.core.world"]:
            sys.modules.pop(mod, None)

        from ecs_agent.core.system import SystemExecutor
        from ecs_agent.core.world import World

        @dataclass(slots=True)
        class FailingSystem:
            name: str

            async def process(self, world: World) -> None:
                raise ValueError("Test error")

        executor = SystemExecutor()
        world = World()
        executor.register(FailingSystem(name="failing_system"), priority=0)

        # System executor should handle exception gracefully
        try:
            await executor.execute(world)
        except Exception:
            pass  # Expected to handle internally

        captured = capsys.readouterr()
        # Parse only JSON lines
        events = []
        for line in captured.out.strip().split("\n"):
            if line.strip() and line.strip().startswith("{"):
                events.append(json.loads(line))
        system_error_events = [
            e for e in events if e.get("event") == STANDARD_EVENT_NAMES["SYSTEM_ERROR"]
        ]

        assert len(system_error_events) >= 1
        event = system_error_events[0]
        assert "system" in event
        assert "exception" in event
        assert "Test error" in event["exception"]


@pytest.mark.asyncio
async def test_system_executor_register_system_returns_handle() -> None:
    executor = SystemExecutor()
    handle = executor.register(LoggingSystem(name="single", log=[]), priority=0)

    assert isinstance(handle, str)
    assert handle


@pytest.mark.asyncio
async def test_system_executor_remove_system_uses_handle() -> None:
    executor = SystemExecutor()
    world = World()
    log: list[str] = []

    removed_handle = executor.register(
        LoggingSystem(name="remove_me", log=log), priority=0
    )
    executor.register(LoggingSystem(name="keep_me", log=log), priority=0)
    await executor.execute(world)

    log.clear()
    executor.remove(removed_handle)
    await executor.execute(world)

    assert log == ["keep_me"]


@pytest.mark.asyncio
async def test_system_executor_replace_system_uses_handle_and_keeps_order() -> None:
    executor = SystemExecutor()
    world = World()
    log: list[str] = []

    first_handle = executor.register(LoggingSystem(name="first", log=log), priority=0)
    executor.register(LoggingSystem(name="second", log=log), priority=0)
    await executor.execute(world)
    assert log == ["first", "second"]

    log.clear()
    replacement = LoggingSystem(name="first_replaced", log=log)
    executor.replace(first_handle, replacement, priority=0)
    await executor.execute(world)

    assert log == ["first_replaced", "second"]


@pytest.mark.asyncio
async def test_system_executor_handle_queue_applies_in_deterministic_order() -> None:
    executor = SystemExecutor()
    world = World()
    log: list[str] = []

    first = executor.register(LoggingSystem(name="first", log=log), priority=0)
    second = executor.register(LoggingSystem(name="second", log=log), priority=0)
    await executor.execute(world)
    assert log == ["first", "second"]

    log.clear()
    executor.replace(first, LoggingSystem(name="first_replaced", log=log), priority=0)
    executor.remove(second)
    third = executor.register(LoggingSystem(name="third", log=log), priority=0)
    assert isinstance(third, str)
    await executor.execute(world)

    assert log == ["first_replaced", "third"]


@pytest.mark.asyncio
async def test_system_executor_queued_ops_register_remove_replace_deterministic_order() -> (
    None
):
    executor = SystemExecutor()
    world = World()
    log: list[str] = []

    first = executor.register(LoggingSystem(name="first", log=log), priority=0)
    second = executor.register(LoggingSystem(name="second", log=log), priority=0)
    await executor.execute(world)
    assert log == ["first", "second"]

    log.clear()
    executor.register(LoggingSystem(name="third", log=log), priority=0)
    executor.remove(second)
    executor.replace(first, LoggingSystem(name="first_replaced", log=log), priority=0)
    await executor.execute(world)

    assert log == ["first_replaced", "third"]


@pytest.mark.asyncio
async def test_system_executor_tick_boundary_replace_queued_ops_apply_next_execute() -> (
    None
):
    class QueueReplaceOnceSystem:
        def __init__(
            self,
            executor: SystemExecutor,
            target_handle: SystemHandle,
            log: list[str],
        ) -> None:
            self._executor = executor
            self._target_handle = target_handle
            self._log = log
            self._did_queue = False

        async def process(self, world: World) -> None:
            self._log.append("mutator")
            if self._did_queue:
                return

            self._executor.replace(
                self._target_handle,
                LoggingSystem(name="new", log=self._log),
                priority=1,
            )
            self._did_queue = True

    executor = SystemExecutor()
    world = World()
    log: list[str] = []

    target = executor.register(LoggingSystem(name="old", log=log), priority=1)
    executor.register(
        QueueReplaceOnceSystem(executor=executor, target_handle=target, log=log),
        priority=0,
    )

    await executor.execute(world)
    assert log == ["mutator", "old"]

    log.clear()
    await executor.execute(world)
    assert log == ["mutator", "new"]
