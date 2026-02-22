from dataclasses import dataclass

from ecs_agent.core.component import ComponentStore
from ecs_agent.types import EntityId


@dataclass(slots=True)
class Position:
    x: float
    y: float


@dataclass(slots=True)
class Velocity:
    dx: float
    dy: float


def test_component_store_add_and_get() -> None:
    store = ComponentStore()
    entity_id = EntityId(1)
    component = Position(x=1.0, y=2.0)

    store.add(entity_id, component)
    assert store.get(entity_id, Position) == component


def test_component_store_has_component() -> None:
    store = ComponentStore()
    entity_id = EntityId(1)

    assert not store.has(entity_id, Position)
    store.add(entity_id, Position(x=0.0, y=0.0))
    assert store.has(entity_id, Position)


def test_component_store_remove_component() -> None:
    store = ComponentStore()
    entity_id = EntityId(1)
    store.add(entity_id, Position(x=0.0, y=0.0))

    store.remove(entity_id, Position)
    assert store.get(entity_id, Position) is None


def test_component_store_overwrites_same_component_type() -> None:
    store = ComponentStore()
    entity_id = EntityId(1)

    store.add(entity_id, Position(x=1.0, y=2.0))
    store.add(entity_id, Position(x=3.0, y=4.0))

    result = store.get(entity_id, Position)
    assert result == Position(x=3.0, y=4.0)


def test_component_store_delete_entity_removes_all_component_types() -> None:
    store = ComponentStore()
    entity_id = EntityId(1)

    store.add(entity_id, Position(x=1.0, y=2.0))
    store.add(entity_id, Velocity(dx=0.5, dy=0.25))

    store.delete_entity(entity_id)

    assert store.get(entity_id, Position) is None
    assert store.get(entity_id, Velocity) is None
