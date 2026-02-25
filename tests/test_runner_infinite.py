"""Tests for Runner infinite-loop support (max_ticks=None)."""

import pytest

from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core.runner import Runner
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


class TestRunnerInfiniteLoop:
    """Tests for max_ticks=None (infinite loop until TerminalComponent)."""

    @pytest.fixture
    def world(self) -> World:
        return World()

    @pytest.fixture
    def runner(self) -> Runner:
        return Runner()

    @pytest.mark.asyncio
    async def test_run_max_ticks_none_stops_on_terminal(
        self, world: World, runner: Runner
    ) -> None:
        """max_ticks=None should run until TerminalComponent appears."""
        counter = CounterSystem()
        terminator = TerminateAtTickSystem(terminate_at_tick=7)
        world.register_system(counter, priority=0)
        world.register_system(terminator, priority=1)

        await runner.run(world, max_ticks=None)

        assert counter.run_count == 7
        assert terminator.tick_count == 7

    @pytest.mark.asyncio
    async def test_run_max_ticks_none_no_max_ticks_terminal(
        self, world: World, runner: Runner
    ) -> None:
        """max_ticks=None should NOT create a max_ticks TerminalComponent."""
        counter = CounterSystem()
        terminator = TerminateAtTickSystem(terminate_at_tick=3)
        world.register_system(counter, priority=0)
        world.register_system(terminator, priority=1)

        await runner.run(world, max_ticks=None)

        terminals = list(world.query(TerminalComponent))
        assert len(terminals) == 1
        _, (comp,) = terminals[0]
        assert comp.reason == "test_termination"

    @pytest.mark.asyncio
    async def test_run_max_ticks_none_immediate_terminal(
        self, world: World, runner: Runner
    ) -> None:
        """max_ticks=None with system that terminates on tick 1."""
        counter = CounterSystem()
        terminator = TerminateAtTickSystem(terminate_at_tick=1)
        world.register_system(counter, priority=0)
        world.register_system(terminator, priority=1)

        await runner.run(world, max_ticks=None)

        assert counter.run_count == 1

    @pytest.mark.asyncio
    async def test_run_default_max_ticks_unchanged(
        self, world: World, runner: Runner
    ) -> None:
        """Default max_ticks=100 still works (backward compat)."""
        counter = CounterSystem()
        world.register_system(counter, priority=0)
        world.create_entity()

        await runner.run(world)

        assert counter.run_count == 100

    @pytest.mark.asyncio
    async def test_run_explicit_max_ticks_still_works(
        self, world: World, runner: Runner
    ) -> None:
        """Explicit max_ticks=5 still adds TerminalComponent(reason='max_ticks')."""
        counter = CounterSystem()
        world.register_system(counter, priority=0)
        world.create_entity()

        await runner.run(world, max_ticks=5)

        assert counter.run_count == 5
        terminals = list(world.query(TerminalComponent))
        assert len(terminals) == 1
        _, (comp,) = terminals[0]
        assert comp.reason == "max_ticks"

    @pytest.mark.asyncio
    async def test_run_max_ticks_none_updates_runner_state(
        self, world: World, runner: Runner
    ) -> None:
        """RunnerStateComponent.current_tick should be updated correctly."""
        from ecs_agent.components.definitions import RunnerStateComponent

        terminator = TerminateAtTickSystem(terminate_at_tick=5)
        world.register_system(terminator, priority=0)

        await runner.run(world, max_ticks=None)

        runner_states = list(world.query(RunnerStateComponent))
        assert len(runner_states) == 1
        _, (state,) = runner_states[0]
        assert state.current_tick == 5
