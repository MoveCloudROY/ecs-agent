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
