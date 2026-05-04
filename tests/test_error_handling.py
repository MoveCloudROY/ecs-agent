"""Tests for ErrorHandlingSystem."""

import json
import time

import pytest

from ecs_agent.components.definitions import ErrorComponent
from ecs_agent.logging import configure_logging
from ecs_agent.core.world import World
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.types import ErrorOccurredEvent


def _captured_log_output(captured: object) -> str:
    return str(getattr(captured, "err", "") or getattr(captured, "out", ""))


def _json_events(output: str) -> list[dict[str, object]]:
    """Parse JSON events from logging output."""
    events: list[dict[str, object]] = []
    for line in output.strip().split("\n"):
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip non-JSON lines
                continue
    return events


class TestErrorHandlingSystem:
    """Test ErrorHandlingSystem behavior."""

    @pytest.fixture
    def world(self) -> World:
        """Create a fresh World instance."""
        return World()

    @pytest.fixture
    def system(self) -> ErrorHandlingSystem:
        """Create ErrorHandlingSystem instance."""
        return ErrorHandlingSystem()

    def test_constructor_default_priority(self) -> None:
        """Test ErrorHandlingSystem has default priority of 99."""
        system = ErrorHandlingSystem()
        assert system.priority == 99

    def test_constructor_custom_priority(self) -> None:
        """Test ErrorHandlingSystem accepts custom priority."""
        system = ErrorHandlingSystem(priority=50)
        assert system.priority == 50

    @pytest.mark.asyncio
    async def test_process_queries_error_components(
        self, world: World, system: ErrorHandlingSystem
    ) -> None:
        """Test that process queries entities with ErrorComponent."""
        entity_id = world.create_entity()
        error = ErrorComponent(
            error="Test error", system_name="TestSystem", timestamp=time.time()
        )
        world.add_component(entity_id, error)

        # Process should not crash
        await system.process(world)

    @pytest.mark.asyncio
    async def test_process_removes_error_component(
        self, world: World, system: ErrorHandlingSystem
    ) -> None:
        """Test that process removes ErrorComponent from entity."""
        entity_id = world.create_entity()
        error = ErrorComponent(
            error="Test error", system_name="TestSystem", timestamp=time.time()
        )
        world.add_component(entity_id, error)

        await system.process(world)

        # ErrorComponent should be removed
        assert not world.has_component(entity_id, ErrorComponent)

    @pytest.mark.asyncio
    async def test_process_publishes_error_occurred_event(
        self, world: World, system: ErrorHandlingSystem
    ) -> None:
        """Test that process publishes ErrorOccurredEvent."""
        entity_id = world.create_entity()
        error_msg = "Database connection failed"
        system_name = "DatabaseSystem"
        error = ErrorComponent(
            error=error_msg, system_name=system_name, timestamp=time.time()
        )
        world.add_component(entity_id, error)

        # Subscribe to ErrorOccurredEvent
        events_received = []

        async def event_handler(event: ErrorOccurredEvent) -> None:
            events_received.append(event)

        world.event_bus.subscribe(ErrorOccurredEvent, event_handler)

        await system.process(world)

        # Should have published one event
        assert len(events_received) == 1
        event = events_received[0]
        assert event.entity_id == entity_id
        assert event.error == error_msg
        assert event.system_name == system_name

    @pytest.mark.asyncio
    async def test_process_handles_multiple_errors(
        self, world: World, system: ErrorHandlingSystem
    ) -> None:
        """Test that process handles multiple entities with ErrorComponent."""
        entity1 = world.create_entity()
        entity2 = world.create_entity()
        entity3 = world.create_entity()

        world.add_component(
            entity1,
            ErrorComponent(
                error="Error 1", system_name="System1", timestamp=time.time()
            ),
        )
        world.add_component(
            entity2,
            ErrorComponent(
                error="Error 2", system_name="System2", timestamp=time.time()
            ),
        )
        world.add_component(
            entity3,
            ErrorComponent(
                error="Error 3", system_name="System3", timestamp=time.time()
            ),
        )

        events_received = []

        async def event_handler(event: ErrorOccurredEvent) -> None:
            events_received.append(event)

        world.event_bus.subscribe(ErrorOccurredEvent, event_handler)

        await system.process(world)

        # Should process all 3 errors
        assert len(events_received) == 3
        # All ErrorComponents should be removed
        assert not world.has_component(entity1, ErrorComponent)
        assert not world.has_component(entity2, ErrorComponent)
        assert not world.has_component(entity3, ErrorComponent)

    @pytest.mark.asyncio
    async def test_process_no_errors_no_crash(
        self, world: World, system: ErrorHandlingSystem
    ) -> None:
        """Test that process handles no errors gracefully."""
        # Create entities without ErrorComponent
        world.create_entity()
        world.create_entity()

        # Should not crash
        await system.process(world)

    @pytest.mark.asyncio
    async def test_process_prints_error_log(
        self,
        world: World,
        system: ErrorHandlingSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that process emits the error log after explicit logging setup."""
        configure_logging(json_output=True, level="ERROR")

        entity_id = world.create_entity()
        error = ErrorComponent(
            error="Critical failure",
            system_name="CriticalSystem",
            timestamp=time.time(),
        )
        world.add_component(entity_id, error)

        await system.process(world)

        captured = capsys.readouterr()
        events = _json_events(_captured_log_output(captured))

        # Find entity_error event
        error_events = [e for e in events if e.get("event") == "entity_error"]
        assert len(error_events) > 0, "No entity_error event found"

        event = error_events[0]
        assert event.get("entity_id") == entity_id
        assert event.get("system_name") == "CriticalSystem"
        assert event.get("error") == "Critical failure"
