from dataclasses import dataclass

import json

import pytest

from ecs_agent.core import EventBus
from ecs_agent.core.world import World
from ecs_agent.types import EntityId


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

@dataclass(slots=True)
class Position:
    x: float
    y: float


@dataclass(slots=True)
class Velocity:
    dx: float
    dy: float


class TrackingSystem:
    def __init__(self, marker: str, log: list[str]) -> None:
        self._marker = marker
        self._log = log

    async def process(self, world: World) -> None:
        _ = world
        self._log.append(self._marker)


def test_world_create_entity_returns_incrementing_ids() -> None:
    world = World()
    first = world.create_entity()
    second = world.create_entity()
    assert first == EntityId(1)
    assert second == EntityId(2)


def test_world_event_bus_property_exposes_event_bus_instance() -> None:
    world = World()
    assert isinstance(world.event_bus, EventBus)


def test_world_add_and_get_component() -> None:
    world = World()
    entity = world.create_entity()
    position = Position(x=1.0, y=2.0)

    world.add_component(entity, position)
    assert world.get_component(entity, Position) == position


def test_world_get_missing_component_returns_none() -> None:
    world = World()
    entity = world.create_entity()

    assert world.get_component(entity, Position) is None


def test_world_has_component_reflects_component_presence() -> None:
    world = World()
    entity = world.create_entity()

    assert not world.has_component(entity, Position)
    world.add_component(entity, Position(x=0.0, y=0.0))
    assert world.has_component(entity, Position)


def test_world_remove_component_deletes_component() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(entity, Position(x=1.0, y=2.0))

    world.remove_component(entity, Position)
    assert not world.has_component(entity, Position)


def test_world_add_component_overwrites_same_type() -> None:
    world = World()
    entity = world.create_entity()

    world.add_component(entity, Position(x=1.0, y=2.0))
    world.add_component(entity, Position(x=3.0, y=4.0))

    assert world.get_component(entity, Position) == Position(x=3.0, y=4.0)


def test_world_delete_entity_removes_all_components() -> None:
    world = World()
    entity = world.create_entity()

    world.add_component(entity, Position(x=1.0, y=2.0))
    world.add_component(entity, Velocity(dx=0.5, dy=1.5))
    world.delete_entity(entity)

    assert world.get_component(entity, Position) is None
    assert world.get_component(entity, Velocity) is None


def test_world_query_returns_expected_components() -> None:
    world = World()
    a = world.create_entity()
    b = world.create_entity()

    world.add_component(a, Position(x=1.0, y=2.0))
    world.add_component(a, Velocity(dx=0.1, dy=0.2))
    world.add_component(b, Position(x=3.0, y=4.0))

    results = world.query(Position, Velocity)
    assert results == [(a, (Position(x=1.0, y=2.0), Velocity(dx=0.1, dy=0.2)))]


@pytest.mark.asyncio
async def test_world_process_executes_systems_by_priority() -> None:
    world = World()
    log: list[str] = []

    world.register_system(TrackingSystem("p1", log), priority=1)
    world.register_system(TrackingSystem("p0", log), priority=0)

    await world.process()
    assert log == ["p0", "p1"]



def test_world_create_entity_logs_entity_created(capsys) -> None:
    """Test create_entity emits entity_created debug event."""
    world = World()
    entity_id = world.create_entity()

    captured = capsys.readouterr()
    output = captured.out

    # Check for entity_created event
    events = _json_events(output)
    entity_created_events = [e for e in events if e.get("event") == "entity_created"]
    assert len(entity_created_events) > 0, "No entity_created event found"
    event = entity_created_events[0]
    assert event.get("entity_id") == entity_id

def test_world_add_component_logs_component_added(capsys) -> None:
    """Test add_component emits component_added info event."""
    world = World()
    entity = world.create_entity()
    position = Position(x=1.0, y=2.0)

    world.add_component(entity, position)

    captured = capsys.readouterr()
    output = captured.out

    # Check for component_added event
    events = _json_events(output)
    component_added_events = [e for e in events if e.get("event") == "component_added"]
    assert len(component_added_events) > 0, "No component_added event found"
    event = component_added_events[0]
    assert event.get("entity_id") == entity
    assert event.get("component_type") == "Position"
    # Verify component payload is not logged
    output_str = json.dumps(output)
    assert "x=" not in output_str
    assert "y=" not in output_str