from dataclasses import dataclass

import json

import pytest

from ecs_agent.core import EventBus
from ecs_agent.core.world import World
from ecs_agent.logging import configure_logging
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
    configure_logging(json_output=True, level="DEBUG")

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
    configure_logging(json_output=True, level="INFO")

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


# =======================
# Entity Registry Tests
# =======================


def test_world_register_entity_adds_to_registry() -> None:
    """Test register_entity adds entity to internal registry."""
    world = World()
    entity = world.create_entity()
    
    world.register_entity(entity, "player", None)
    
    resolved = world.resolve_entity("player")
    assert resolved == entity


def test_world_resolve_entity_returns_correct_id() -> None:
    """Test resolve_entity returns correct entity ID for registered name."""
    world = World()
    entity = world.create_entity()
    
    world.register_entity(entity, "agent_1", None)
    
    assert world.resolve_entity("agent_1") == entity


def test_world_resolve_entity_returns_none_for_missing() -> None:
    """Test resolve_entity returns None for unregistered name."""
    world = World()
    
    assert world.resolve_entity("nonexistent") is None


def test_world_register_entity_duplicate_name_raises_error() -> None:
    """Test register_entity raises ValueError for duplicate names."""
    world = World()
    entity1 = world.create_entity()
    entity2 = world.create_entity()
    
    world.register_entity(entity1, "duplicate", None)
    
    with pytest.raises(ValueError, match="Entity name 'duplicate' already registered"):
        world.register_entity(entity2, "duplicate", None)


def test_world_list_entities_by_tag_returns_matching_ids() -> None:
    """Test list_entities_by_tag returns all entities with specified tag."""
    world = World()
    agent1 = world.create_entity()
    agent2 = world.create_entity()
    npc = world.create_entity()
    
    world.register_entity(agent1, "agent_1", {"ai", "friendly"})
    world.register_entity(agent2, "agent_2", {"ai", "hostile"})
    world.register_entity(npc, "npc_1", {"npc"})
    
    ai_entities = world.list_entities_by_tag("ai")
    assert set(ai_entities) == {agent1, agent2}


def test_world_list_entities_by_tag_returns_empty_for_missing_tag() -> None:
    """Test list_entities_by_tag returns empty list for missing tag."""
    world = World()
    entity = world.create_entity()
    
    world.register_entity(entity, "agent", {"ai"})
    
    result = world.list_entities_by_tag("nonexistent")
    assert result == []


def test_world_delete_entity_removes_from_registry() -> None:
    """Test delete_entity automatically removes entity from registry."""
    world = World()
    entity = world.create_entity()
    
    world.register_entity(entity, "to_delete", {"temp"})
    assert world.resolve_entity("to_delete") == entity
    
    world.delete_entity(entity)
    
    assert world.resolve_entity("to_delete") is None
    assert entity not in world.list_entities_by_tag("temp")


def test_world_unregister_entity_cleans_up_tags() -> None:
    """Test unregister_entity removes entity from all tag indexes."""
    world = World()
    entity = world.create_entity()
    
    world.register_entity(entity, "multi_tag", {"tag1", "tag2", "tag3"})
    assert entity in world.list_entities_by_tag("tag1")
    assert entity in world.list_entities_by_tag("tag2")
    assert entity in world.list_entities_by_tag("tag3")
    
    world.unregister_entity(entity)
    
    assert entity not in world.list_entities_by_tag("tag1")
    assert entity not in world.list_entities_by_tag("tag2")
    assert entity not in world.list_entities_by_tag("tag3")
    assert world.resolve_entity("multi_tag") is None
