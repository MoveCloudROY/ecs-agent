"""Tests for Runner."""

import pytest

from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core.runner import Runner
from ecs_agent.core.system import System
from ecs_agent.core.world import World


class CounterSystem:
    """Test system that counts how many times it runs."""

    def __init__(self, priority: int = 0) -> None:
        self.priority = priority
        self.run_count = 0

    async def process(self, world: World) -> None:
        self.run_count += 1


class TerminateAtTickSystem:
    """Test system that adds TerminalComponent after N ticks."""

    def __init__(self, terminate_at_tick: int, priority: int = 0) -> None:
        self.priority = priority
        self.terminate_at_tick = terminate_at_tick
        self.tick_count = 0

    async def process(self, world: World) -> None:
        self.tick_count += 1
        if self.tick_count >= self.terminate_at_tick:
            entity_id = world.create_entity()
            world.add_component(entity_id, TerminalComponent(reason="test_termination"))


class TestRunner:
    """Test Runner behavior."""

    @pytest.fixture
    def world(self) -> World:
        """Create a fresh World instance."""
        return World()

    @pytest.fixture
    def runner(self) -> Runner:
        """Create Runner instance."""
        return Runner()

    @pytest.mark.asyncio
    async def test_run_processes_world(self, world: World, runner: Runner) -> None:
        """Test that run calls world.process()."""
        counter = CounterSystem()
        world.register_system(counter, priority=0)

        await runner.run(world, max_ticks=5)

        assert counter.run_count > 0

    @pytest.mark.asyncio
    async def test_run_stops_on_terminal_component(
        self, world: World, runner: Runner
    ) -> None:
        """Test that run stops when TerminalComponent is found."""
        counter = CounterSystem()
        terminator = TerminateAtTickSystem(terminate_at_tick=3)
        world.register_system(counter, priority=0)
        world.register_system(terminator, priority=1)

        await runner.run(world, max_ticks=100)

        assert counter.run_count == 3
        assert terminator.tick_count == 3

    @pytest.mark.asyncio
    async def test_run_adds_terminal_on_max_ticks(
        self, world: World, runner: Runner
    ) -> None:
        """Test that run adds TerminalComponent when max_ticks is reached."""
        counter = CounterSystem()
        world.register_system(counter, priority=0)

        entity_id = world.create_entity()

        await runner.run(world, max_ticks=10)

        assert counter.run_count == 10
        has_terminal = any(
            world.has_component(eid, TerminalComponent)
            for eid, _ in world.query(TerminalComponent)
        )
        assert has_terminal

    @pytest.mark.asyncio
    async def test_run_terminal_reason_max_ticks(
        self, world: World, runner: Runner
    ) -> None:
        """Test that TerminalComponent reason is 'max_ticks' when limit reached."""
        counter = CounterSystem()
        world.register_system(counter, priority=0)

        entity_id = world.create_entity()

        await runner.run(world, max_ticks=5)

        terminal_components = list(world.query(TerminalComponent))
        assert len(terminal_components) == 1
        _, (terminal_comp,) = terminal_components[0]
        assert terminal_comp.reason == "max_ticks"

    @pytest.mark.asyncio
    async def test_run_default_max_ticks(self, world: World, runner: Runner) -> None:
        """Test that run uses default max_ticks=100 when not specified."""
        counter = CounterSystem()
        world.register_system(counter, priority=0)

        entity_id = world.create_entity()

        await runner.run(world)

        assert counter.run_count == 100

    @pytest.mark.asyncio
    async def test_run_no_crash_empty_world(self, world: World, runner: Runner) -> None:
        """Test that run handles empty world gracefully."""
        await runner.run(world, max_ticks=5)

    @pytest.mark.asyncio
    async def test_run_immediate_terminal(self, world: World, runner: Runner) -> None:
        """Test that run stops immediately if TerminalComponent exists before first tick."""
        counter = CounterSystem()
        terminator = TerminateAtTickSystem(terminate_at_tick=1)
        world.register_system(counter, priority=0)
        world.register_system(terminator, priority=1)

        await runner.run(world, max_ticks=100)

        assert counter.run_count == 1
        assert terminator.tick_count == 1

    @pytest.mark.asyncio
    async def test_run_multiple_ticks(self, world: World, runner: Runner) -> None:
        """Test that each run call processes exactly max_ticks times."""
        counter = CounterSystem()
        world.register_system(counter, priority=0)

        entity_id = world.create_entity()

        await runner.run(world, max_ticks=7)

        assert counter.run_count == 7


class TestRunnerLogging:
    """Test Runner lifecycle logging."""


    @pytest.mark.asyncio
    async def test_runner_emits_run_start_event(self, capsys) -> None:
        """Test that runner emits run_start event at beginning."""
        import json
        import sys
        from ecs_agent.logging import configure_logging, STANDARD_EVENT_NAMES

        configure_logging(json_output=True, level="INFO")

        # Force reimport to pick up new logger config
        for mod in ["ecs_agent.core.runner", "ecs_agent.core.world", "ecs_agent.core.system"]:
            sys.modules.pop(mod, None)

        from ecs_agent.core.world import World
        from ecs_agent.core.runner import Runner

        world = World()
        runner = Runner()
        counter = CounterSystem()
        world.register_system(counter, priority=0)
        await runner.run(world, max_ticks=1)

        captured = capsys.readouterr()
        # Parse only JSON lines (filter out console-formatted debug lines from pre-configured modules)
        events = []
        for line in captured.out.strip().split("\n"):
            if line.strip() and line.strip().startswith("{"):
                events.append(json.loads(line))
        run_start_events = [e for e in events if e.get("event") == STANDARD_EVENT_NAMES["RUN_START"]]

        assert len(run_start_events) >= 1
        event = run_start_events[0]
        assert "max_ticks" in event
        assert event["max_ticks"] == 1

    @pytest.mark.asyncio
    async def test_runner_emits_run_complete_event(self, capsys) -> None:
        """Test that runner emits run_complete event at end."""
        import json
        import sys
        from ecs_agent.logging import configure_logging, STANDARD_EVENT_NAMES

        configure_logging(json_output=True, level="INFO")

        # Force reimport to pick up new logger config
        for mod in ["ecs_agent.core.runner", "ecs_agent.core.world", "ecs_agent.core.system"]:
            sys.modules.pop(mod, None)

        from ecs_agent.core.world import World
        from ecs_agent.core.runner import Runner

        world = World()
        runner = Runner()
        counter = CounterSystem()
        world.register_system(counter, priority=0)

        await runner.run(world, max_ticks=2)

        captured = capsys.readouterr()
        # Parse only JSON lines
        events = []
        for line in captured.out.strip().split("\n"):
            if line.strip() and line.strip().startswith("{"):
                events.append(json.loads(line))
        run_complete_events = [e for e in events if e.get("event") == STANDARD_EVENT_NAMES["RUN_COMPLETE"]]

        assert len(run_complete_events) >= 1
        event = run_complete_events[0]
        assert "reason" in event

    @pytest.mark.asyncio
    async def test_runner_emits_tick_start_and_complete_events(self, capsys) -> None:
        """Test that runner emits tick_start and tick_complete for each tick."""
        import json
        import sys
        from ecs_agent.logging import configure_logging, STANDARD_EVENT_NAMES

        configure_logging(json_output=True, level="DEBUG")

        # Force reimport to pick up new logger config
        for mod in ["ecs_agent.core.runner", "ecs_agent.core.world", "ecs_agent.core.system"]:
            sys.modules.pop(mod, None)

        from ecs_agent.core.world import World
        from ecs_agent.core.runner import Runner

        world = World()
        runner = Runner()
        counter = CounterSystem()
        world.register_system(counter, priority=0)

        await runner.run(world, max_ticks=3)

        captured = capsys.readouterr()
        # Parse only JSON lines
        events = []
        for line in captured.out.strip().split("\n"):
            if line.strip() and line.strip().startswith("{"):
                events.append(json.loads(line))
        tick_start_events = [e for e in events if e.get("event") == STANDARD_EVENT_NAMES["TICK_START"]]
        tick_complete_events = [e for e in events if e.get("event") == STANDARD_EVENT_NAMES["TICK_COMPLETE"]]

        # Should have 3 tick_start and 3 tick_complete events
        assert len(tick_start_events) == 3
        assert len(tick_complete_events) == 3

        # Verify tick numbers are sequential
        for i, event in enumerate(tick_start_events):
            assert "tick" in event
            assert event["tick"] == i

    @pytest.mark.asyncio
    async def test_runner_tick_complete_includes_duration_ms(self, capsys) -> None:
        """Test that tick_complete events include duration_ms field."""
        import json
        import sys
        from ecs_agent.logging import configure_logging, STANDARD_EVENT_NAMES

        configure_logging(json_output=True, level="DEBUG")

        # Force reimport to pick up new logger config
        for mod in ["ecs_agent.core.runner", "ecs_agent.core.world", "ecs_agent.core.system"]:
            sys.modules.pop(mod, None)

        from ecs_agent.core.world import World
        from ecs_agent.core.runner import Runner

        world = World()
        runner = Runner()
        counter = CounterSystem()
        world.register_system(counter, priority=0)

        await runner.run(world, max_ticks=1)

        captured = capsys.readouterr()
        # Parse only JSON lines
        events = []
        for line in captured.out.strip().split("\n"):
            if line.strip() and line.strip().startswith("{"):
                events.append(json.loads(line))
        tick_complete_events = [e for e in events if e.get("event") == STANDARD_EVENT_NAMES["TICK_COMPLETE"]]

        assert len(tick_complete_events) >= 1
        event = tick_complete_events[0]
        assert "duration_ms" in event
        assert isinstance(event["duration_ms"], (int, float))
        assert event["duration_ms"] >= 0
