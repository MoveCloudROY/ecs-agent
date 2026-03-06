"""Tests for Runner."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from ecs_agent.components.definitions import (
    ConversationComponent,
    ErrorComponent,
    InterruptionComponent,
    LLMComponent,
    RunnerStateComponent,
    StreamingComponent,
    TerminalComponent,
)
from ecs_agent.core.runner import Runner
from ecs_agent.core.world import World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import (
    CompletionResult,
    EntityId,
    InterruptionReason,
    Message,
    StreamDelta,
)
from ecs_agent.types import SystemHandle


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


class TickAwareLoggingSystem:
    def __init__(self, name: str, log: list[str]) -> None:
        self._name = name
        self._log = log

    async def process(self, world: World) -> None:
        runner_state_entities = list(world.query(RunnerStateComponent))
        _, (runner_state,) = runner_state_entities[0]
        self._log.append(f"{self._name}:{runner_state.current_tick}")


class ReplaceTargetSystem:
    def __init__(self, target: SystemHandle, replacement_log: list[str]) -> None:
        self._target = target
        self._replacement_log = replacement_log
        self._has_replaced = False

    async def process(self, world: World) -> None:
        if self._has_replaced:
            return
        world.replace_system(
            self._target,
            TickAwareLoggingSystem(name="new", log=self._replacement_log),
            priority=1,
        )
        self._has_replaced = True


class RemoveTargetSystem:
    def __init__(self, target: SystemHandle) -> None:
        self._target = target
        self._has_removed = False

    async def process(self, world: World) -> None:
        if self._has_removed:
            return
        world.remove_system(self._target)
        self._has_removed = True


class CancelledStreamingFakeProvider(FakeProvider):
    async def _stream_complete(
        self, result: CompletionResult
    ) -> AsyncIterator[StreamDelta]:
        _ = result
        yield StreamDelta(content="partial")
        raise asyncio.CancelledError()


class InterruptOneEntitySystem:
    def __init__(self, target_entity: EntityId) -> None:
        self._target_entity = target_entity
        self._interrupted = False

    async def process(self, world: World) -> None:
        if self._interrupted:
            return
        world.add_component(
            self._target_entity,
            InterruptionComponent(reason=InterruptionReason.SYSTEM_PAUSE),
        )
        self._interrupted = True


class CountUninterruptedEntitiesSystem:
    def __init__(self) -> None:
        self.active_ticks = 0

    async def process(self, world: World) -> None:
        active_entities = [
            entity_id
            for entity_id, _ in world.query(ConversationComponent)
            if not world.has_component(entity_id, InterruptionComponent)
        ]

        if active_entities:
            self.active_ticks += 1

        if self.active_ticks >= 3:
            world.add_component(
                world.create_entity(),
                TerminalComponent(reason="uninterrupted_complete"),
            )


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

        world.create_entity()

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

        world.create_entity()

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

        world.create_entity()

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

        world.create_entity()

        await runner.run(world, max_ticks=7)

        assert counter.run_count == 7

    @pytest.mark.asyncio
    async def test_tick_boundary_system_replace_does_not_mutate_current_tick(
        self, world: World, runner: Runner
    ) -> None:
        log: list[str] = []

        replaced_handle = world.register_system(
            TickAwareLoggingSystem(name="old", log=log), priority=1
        )
        world.register_system(
            ReplaceTargetSystem(target=replaced_handle, replacement_log=log), priority=0
        )

        await runner.run(world, max_ticks=2)

        assert log == ["old:0", "new:1"]

    @pytest.mark.asyncio
    async def test_tick_boundary_system_remove_applies_on_next_tick(
        self, world: World, runner: Runner
    ) -> None:
        log: list[str] = []

        removed_handle = world.register_system(
            TickAwareLoggingSystem(name="victim", log=log), priority=1
        )
        world.register_system(RemoveTargetSystem(target=removed_handle), priority=0)

        await runner.run(world, max_ticks=2)

        assert log == ["victim:0"]

    @pytest.mark.asyncio
    async def test_runner_system_replace_next_tick_applies_pending_pre_tick(
        self, runner: Runner
    ) -> None:
        class OrderedApplyWorld(World):
            def __init__(self) -> None:
                super().__init__()
                self.call_order: list[str] = []

            def apply_pending_system_operations(self) -> None:
                self.call_order.append("apply_pending")
                super().apply_pending_system_operations()

            async def process(self) -> None:
                self.call_order.append("process")
                terminal_entity = self.create_entity()
                self.add_component(terminal_entity, TerminalComponent(reason="done"))

        world = OrderedApplyWorld()

        await runner.run(world, max_ticks=2)

        assert world.call_order == ["apply_pending", "process"]

    @pytest.mark.asyncio
    async def test_runner_graceful_interrupt_preserves_partial(
        self, world: World, runner: Runner
    ) -> None:
        provider = CancelledStreamingFakeProvider(
            responses=[
                CompletionResult(message=Message(role="assistant", content="ignored"))
            ]
        )
        entity_id = world.create_entity()
        world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
        world.add_component(
            entity_id,
            ConversationComponent(messages=[Message(role="user", content="Hello")]),
        )
        world.add_component(entity_id, StreamingComponent(enabled=True))
        world.register_system(ReasoningSystem(), priority=0)

        await runner.run(world, max_ticks=5)

        conversation = world.get_component(entity_id, ConversationComponent)
        interruption = world.get_component(entity_id, InterruptionComponent)
        assert conversation is not None
        assert conversation.messages[-1] == Message(role="assistant", content="partial")
        assert interruption is not None

    @pytest.mark.asyncio
    async def test_runner_cancelled_error_not_misclassified(
        self, world: World, runner: Runner
    ) -> None:
        provider = CancelledStreamingFakeProvider(
            responses=[
                CompletionResult(message=Message(role="assistant", content="ignored"))
            ]
        )
        entity_id = world.create_entity()
        world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
        world.add_component(
            entity_id,
            ConversationComponent(messages=[Message(role="user", content="Hello")]),
        )
        world.add_component(entity_id, StreamingComponent(enabled=True))
        world.register_system(ReasoningSystem(), priority=0)

        await runner.run(world, max_ticks=5)

        assert world.get_component(entity_id, ErrorComponent) is None
        assert world.get_component(entity_id, InterruptionComponent) is not None

    @pytest.mark.asyncio
    async def test_interruption_component_attached_on_graceful_stop(
        self, world: World, runner: Runner
    ) -> None:
        provider = CancelledStreamingFakeProvider(
            responses=[
                CompletionResult(message=Message(role="assistant", content="ignored"))
            ]
        )
        entity_id = world.create_entity()
        world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
        world.add_component(
            entity_id,
            ConversationComponent(messages=[Message(role="user", content="Hello")]),
        )
        world.add_component(entity_id, StreamingComponent(enabled=True))
        world.register_system(ReasoningSystem(), priority=0)

        await runner.run(world, max_ticks=5)

        interruption = world.get_component(entity_id, InterruptionComponent)
        assert interruption is not None
        assert interruption.reason == InterruptionReason.USER_REQUESTED

    @pytest.mark.asyncio
    async def test_runner_continues_after_entity_interrupted(
        self, world: World, runner: Runner
    ) -> None:
        interrupted_entity = world.create_entity()
        active_entity = world.create_entity()

        world.add_component(
            interrupted_entity,
            ConversationComponent(messages=[Message(role="user", content="paused")]),
        )
        world.add_component(
            active_entity,
            ConversationComponent(messages=[Message(role="user", content="active")]),
        )

        world.register_system(
            InterruptOneEntitySystem(target_entity=interrupted_entity), priority=0
        )
        counter = CountUninterruptedEntitiesSystem()
        world.register_system(counter, priority=1)

        await runner.run(world, max_ticks=10)

        assert counter.active_ticks == 3


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
        for mod in [
            "ecs_agent.core.runner",
            "ecs_agent.core.world",
            "ecs_agent.core.system",
        ]:
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
        run_start_events = [
            e for e in events if e.get("event") == STANDARD_EVENT_NAMES["RUN_START"]
        ]

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
        for mod in [
            "ecs_agent.core.runner",
            "ecs_agent.core.world",
            "ecs_agent.core.system",
        ]:
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
        run_complete_events = [
            e for e in events if e.get("event") == STANDARD_EVENT_NAMES["RUN_COMPLETE"]
        ]

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
        for mod in [
            "ecs_agent.core.runner",
            "ecs_agent.core.world",
            "ecs_agent.core.system",
        ]:
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
        tick_start_events = [
            e for e in events if e.get("event") == STANDARD_EVENT_NAMES["TICK_START"]
        ]
        tick_complete_events = [
            e for e in events if e.get("event") == STANDARD_EVENT_NAMES["TICK_COMPLETE"]
        ]

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
        for mod in [
            "ecs_agent.core.runner",
            "ecs_agent.core.world",
            "ecs_agent.core.system",
        ]:
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
        tick_complete_events = [
            e for e in events if e.get("event") == STANDARD_EVENT_NAMES["TICK_COMPLETE"]
        ]

        assert len(tick_complete_events) >= 1
        event = tick_complete_events[0]
        assert "duration_ms" in event
        assert isinstance(event["duration_ms"], (int, float))
        assert event["duration_ms"] >= 0
