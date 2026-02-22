from dataclasses import dataclass

from ecs_agent.core.component import ComponentStore
from ecs_agent.core.query import Query
from ecs_agent.types import EntityId


@dataclass(slots=True)
class Position:
    x: float
    y: float


@dataclass(slots=True)
class Velocity:
    dx: float
    dy: float


@dataclass(slots=True)
class Health:
    hp: int


def test_query_single_component_returns_matching_entities() -> None:
    store = ComponentStore()
    query = Query(store)

    store.add(EntityId(1), Position(x=1.0, y=2.0))
    store.add(EntityId(2), Position(x=3.0, y=4.0))

    results = query.get(Position)
    assert results == [
        (EntityId(1), (Position(x=1.0, y=2.0),)),
        (EntityId(2), (Position(x=3.0, y=4.0),)),
    ]


def test_query_multi_component_requires_all_components() -> None:
    store = ComponentStore()
    query = Query(store)

    store.add(EntityId(1), Position(x=1.0, y=2.0))
    store.add(EntityId(1), Velocity(dx=0.1, dy=0.2))
    store.add(EntityId(2), Position(x=3.0, y=4.0))

    results = query.get(Position, Velocity)
    assert results == [
        (EntityId(1), (Position(x=1.0, y=2.0), Velocity(dx=0.1, dy=0.2)))
    ]


def test_query_returns_empty_for_missing_component_type() -> None:
    store = ComponentStore()
    query = Query(store)

    store.add(EntityId(1), Position(x=1.0, y=2.0))
    results = query.get(Velocity)

    assert results == []


def test_query_returns_empty_when_no_entities_match_full_signature() -> None:
    store = ComponentStore()
    query = Query(store)

    store.add(EntityId(1), Position(x=1.0, y=2.0))
    store.add(EntityId(2), Velocity(dx=0.1, dy=0.2))

    results = query.get(Position, Velocity, Health)
    assert results == []


def test_query_preserves_component_order_in_result_tuples() -> None:
    store = ComponentStore()
    query = Query(store)

    store.add(EntityId(1), Position(x=1.0, y=2.0))
    store.add(EntityId(1), Velocity(dx=0.1, dy=0.2))

    results = query.get(Velocity, Position)
    assert results == [
        (EntityId(1), (Velocity(dx=0.1, dy=0.2), Position(x=1.0, y=2.0)))
    ]
