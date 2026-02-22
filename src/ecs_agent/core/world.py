from __future__ import annotations

from typing import Any, TypeVar

from ecs_agent.core.component import ComponentStore
from ecs_agent.core.entity import EntityIdGenerator
from ecs_agent.core.event_bus import EventBus
from ecs_agent.core.query import Query
from ecs_agent.core.system import System, SystemExecutor
from ecs_agent.types import EntityId

T = TypeVar("T")


class World:
    def __init__(self) -> None:
        self._entity_gen = EntityIdGenerator()
        self._components = ComponentStore()
        self._systems = SystemExecutor()
        self._event_bus = EventBus()
        self._query = Query(self._components)

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    def create_entity(self) -> EntityId:
        return self._entity_gen.next()

    def add_component(self, entity_id: EntityId, component: Any) -> None:
        self._components.add(entity_id, component)

    def get_component(self, entity_id: EntityId, component_type: type[T]) -> T | None:
        return self._components.get(entity_id, component_type)

    def remove_component(self, entity_id: EntityId, component_type: type[Any]) -> None:
        self._components.remove(entity_id, component_type)

    def has_component(self, entity_id: EntityId, component_type: type[Any]) -> bool:
        return self._components.has(entity_id, component_type)

    def delete_entity(self, entity_id: EntityId) -> None:
        self._components.delete_entity(entity_id)

    def register_system(self, system: System, priority: int) -> None:
        self._systems.register(system, priority)

    async def process(self) -> None:
        await self._systems.execute(self)

    def query(
        self, *component_types: type[Any]
    ) -> list[tuple[EntityId, tuple[Any, ...]]]:
        return self._query.get(*component_types)
